import os
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from .utils.lshash_torch import LSHash
import torch
import torch.nn.functional as F

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_repellency_method(name: str, ref_data, embed_fn, n_embed, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](ref_data, embed_fn, n_embed, **kwargs)

class RepellencyMethod(ABC):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        self.ref_data = ref_data
        self.embed_fn = embed_fn
        self.n_embed = n_embed
        self.scale = kwargs.get('scale', 1.0)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        self.proj_ref_path = kwargs.get('proj_ref_path', None)
        self.cache_proj_ref = kwargs.get('cache_proj_ref', False)
        
        if self.cache_proj_ref:
            self.proj_refs = self.import_proj_ref(self.proj_ref_path)
        else:
            self.proj_refs = self.set_proj_ref()
        
    @torch.no_grad()
    def project(self, data, **kwargs):
        results = []
        if len(data) > self.n_embed:
            for i in range(0, len(data), self.n_embed):
                temp = self.embed_fn(data[i:min(i + self.n_embed, len(data))])
                results.append(temp)
            results = torch.cat(results, 0)
            
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            
            # type casting
            results = results.float()
            return results
        else:
            results = self.embed_fn(data)
            
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            
            # type casting
            results = results.float()
            return results
        
    def discrete_to_continous_time(self, idx, **kwargs):
        if idx == 0:
            result = 0.001
        else:
            result = idx / self.max_idx
        return result
    
    def sigma_edm(self, cont_time, **kwargs):
        return torch.sqrt(torch.exp(0.5*self.beta_max*cont_time**2 + self.beta_min * cont_time) - 1.)
    
    def sigma(self, cont_time, **kwargs):
        pass
        
    def mkdir_cache(self):
        dir_path = os.path.split(self.proj_ref_path)[0]
        os.makedirs(dir_path, exist_ok=True)
        
    def set_proj_ref(self):
        result = []
        with torch.no_grad():
            emb = self.project(self.ref_data)
            result.append(emb.cpu())

            # del
            del emb
            torch.cuda.empty_cache()
        
        # Save the result
        result = torch.cat(result, 0)
        print("[Proj_Ref] Save the cached proj_ref")
        self.mkdir_cache()
        torch.save(result, self.proj_ref_path)

        # map location (cuda)
        result = result.to('cuda')
        return result

    def import_proj_ref(self, proj_ref_path):
        result = torch.load(proj_ref_path, map_location=self.ref_data.device)
        return result
        
    def get_proj_ref(self):
        return self.proj_refs
    
    @abstractmethod
    def empirical_denoiser(self, data, **kwargs):
        pass
   
    def conditioning(self, x_0_hat, **kwargs):
        if x_0_hat.dtype != self.ref_data.dtype:
            x_0_hat = x_0_hat.to(self.ref_data.dtype)
        
        if kwargs.get("guidance_scale", None) is not None and kwargs.get("guidance_scale", None) > 0.0:
            return self.conditioning_2(x_0_hat, **kwargs)
        else:
            return self.conditioning_1(x_0_hat, **kwargs)
            
    def conditioning_1(self, x_0_hat, **kwargs):
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, **kwargs)
        x_0_hat -= self.scale * negative_x_0_hat
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item}

@register_conditioning_method(name='safe_denoiser')
class RBFKernelRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, embed_fn, n_embed, **kwargs)
        self.scale = kwargs.get('scale', 1.0)
        
    def empirical_denoiser(self, x_t, sigma=1.0, **kwargs):
        '''
            args:
                x_t: latent (no need to be embedded)
            returns:
                negative_score: negative score
                negative_score_item: mean of negative score
        '''
        
        # Embedding for kernel distance
        x_t_proj = x_t.reshape(x_t.shape[0], -1)
        ref_data_latent = self.get_proj_ref()
        ref_data_proj = ref_data_latent.reshape(ref_data_latent.shape[0], -1)
        
        # reshape
        N, D = x_t_proj.shape
        M, D = ref_data_proj.shape
        _, C, H, W = ref_data_latent.shape
        x_t_proj = x_t_proj.reshape(1, -1, D)
        ref_data_proj = ref_data_proj.reshape(1, -1, D)
        
        # data augmentation for denominator
        ref_data_org = ref_data_latent.reshape(M, -1)
        ones = torch.ones(M)[..., None].to(ref_data_org.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_org, ones), dim=1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        # Avoid dividing by zero or very small numbers
        denominator = kernel[:, -1].reshape(-1, 1) + self.epsilon  # Add epsilon to prevent division by zero
        nominator = kernel[:, :-1]

        # Compute negative_score with enhanced stability
        negative_score = nominator / denominator
        negative_score_item = negative_score.clamp(min=-1e10, max=1e10).mean().item()
        
        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item    
    
