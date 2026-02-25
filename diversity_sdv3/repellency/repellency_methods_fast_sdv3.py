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
                # exceptional treatment for device, dtype
                if data.device != "cuda" or data.dtype != torch.half:
                    data_i = data[i:min(i + self.n_embed, len(data))]
                    data_i = data_i.to("cuda", dtype=torch.half)
                    temp = self.embed_fn(data_i)
                else:
                    temp = self.embed_fn(data[i:min(i + self.n_embed, len(data))])
                results.append(temp)
            results = torch.cat(results, 0)
            
            '''
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            '''
            
            # type casting
            results = results.float()

            # del
            del temp
            torch.cuda.empty_cache()

            return results
        else:
            results = self.embed_fn(data)
            
            '''
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            '''
            
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
        return self.update(x_0_hat, **kwargs)
            
    def update(self, x_0_hat, **kwargs):
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, **kwargs)
        x_0_hat += self.scale * negative_x_0_hat
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item}
            
@register_conditioning_method(name='grad_mmd')
class GradMMDRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, embed_fn, n_embed, **kwargs)
        self.scale = kwargs.get('scale', 1.0)
        self.normalize = kwargs.get('normalize', False)
        self.max_norm = kwargs.get('max_norm', None) # None or float

    def empirical_denoiser(self, x_t, sigma=-1.0, **kwargs):
        # grad_mmd는 수치 미분이 아니라 닫힌형식 기울기를 반환하므로 autograd 불필요
        dK_dX, dK_dX_stats = self.grad_mmd(x_t, sigma=sigma)

        # 안정화 옵션
        if self.normalize:
            eps = getattr(self, 'epsilon', 1e-8)
            nrm = dK_dX.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(eps).view(-1,1,1,1)
            dK_dX = dK_dX / nrm

        if self.max_norm is not None:
            if self.max_norm is not None: max_norm = 1
            
            eps = getattr(self, 'epsilon', 1e-8)
            nrm = dK_dX.flatten(1).norm(p=2, dim=1, keepdim=True)
            scale = (max_norm / nrm.clamp_min(eps)).clamp(max=1.0).view(-1,1,1,1)
            dK_dX = dK_dX * scale

        return dK_dX, float(dK_dX_stats)

    @torch.no_grad()
    def grad_mmd(self, x0t, sigma=-1.0):
        K, dK_dX = self.rbf_kernel(x0t, self.proj_refs, gamma=sigma)
        return dK_dX, dK_dX.mean().item()

    @torch.no_grad()
    def rbf_kernel(self, X, Y, gamma=-1, ad=1):
        # X and Y should be tensors with shape (batch_size, num_channels, height, width)
        # gamma is a hyperparameter controlling the width of the RBF kernel

        # Reshape X and Y to have shape (batch_size, num_channels*height*width)
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # Compute the pairwise squared Euclidean distances between the samples
        with torch.cuda.amp.autocast():
            dists = torch.cdist(X_flat, Y_flat, p=2)**2
        
        '''
        # use median trick
        if gamma < 0: 
            gamma = torch.median(dists)
            gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
            gamma = 1 / (2 * gamma**2)
            # print(gamma)
        

        # use median trick
        if gamma < 0: 
            k = 20                       # '매우 유사'로 간주할 이웃 수
            sorted_d, _ = torch.sort(dists, dim=1)
            r1_to_k = sorted_d[:, 1:k+1].reshape(-1) 
            gamma = torch.median(r1_to_k)
            gamma = torch.sqrt(0.5 * gamma / np.log(dists.size(0) + 1))
            gamma = 1 / (2 * gamma**2)
            # print(gamma)
        '''

        # use top-k
        if gamma < 0:
            # dists: (N,N) 행렬, 원소는 ‖x_i-x_j‖²
            # self-distance(=0) 제외 후, 각 행마다 k+1개 중 k번째 거리 선택
            k = 3                       # '매우 유사'로 간주할 이웃 수
            sorted_d, _ = torch.sort(dists, dim=1)
            # r_k = sorted_d[:, k]        # 각 샘플에서 k-번째로 작은 제곱거리
            r1_to_k = sorted_d[:, 3:k+3].reshape(-1) 
            r_k2 = r1_to_k.mean()           # 전체 평균을 전역 bandwidth 로 사용
            eps = 0.05                 # 0.05 ~ 0.2 사이에서 실험 권장
            gamma = -torch.log(torch.tensor(eps)) / r_k2
        
        gamma = gamma * ad 
        # gamma = torch.max(gamma, torch.tensor(1e-3))
        
        # Compute the RBF kernel using the squared distances and gamma
        K = torch.exp(-gamma * dists)
        # dK = -2 * gamma * K.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0)) # Out of memory
        dK_dX = self.batched_dK_dX(X_flat, Y_flat, gamma, batch_size=1000)

        # dK_dX의 shape이 X와 다를 경우, 원래 X shape으로 reshape
        if dK_dX.shape != X.shape:
            dK_dX = dK_dX.view(*X.shape)
        
        return K, dK_dX

    @torch.no_grad()
    def batched_dK_dX(self, X, Y, gamma, batch_size=1000):
        N, D = X.shape  # (N, D)
        M, D = Y.shape  # (M, D)
        device = X.device
        dK_dX = torch.zeros_like(X, device=device)

        # batch-wise computation to avoid OOM
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            X_batch = X[i:end_i]  # (batch_size_x, D)

            # Compute pairwise squared distances (batch_size_x, M)
            dists_batch = torch.cdist(X_batch, Y, p=2)**2  # (batch_size_x, M)

            # Compute partial kernel matrix (batch_size_x, M)
            K_batch = torch.exp(-gamma * dists_batch)  # (batch_size_x, M)

            # Compute difference: expand for broadcasting
            X_batch_exp = X_batch.unsqueeze(1)  # (batch_size_x, 1, D)
            Y_exp = Y.unsqueeze(0)              # (1, M, D)

            diff_batch = X_batch_exp - Y_exp    # (batch_size_x, M, D)

            # Reshape K_batch for broadcasting
            K_batch_exp = K_batch.unsqueeze(-1)  # (batch_size_x, M, 1)

            # Compute dK_batch
            dK_batch = -2 * gamma * K_batch_exp * diff_batch  # (batch_size_x, M, D)

            # Sum over Y particles (dimension=1)
            dK_dX_batch = dK_batch.sum(dim=1)  # (batch_size_x, D)

            # Assign computed gradient to output
            dK_dX[i:end_i] = dK_dX_batch

            # Optional: clear GPU memory
            del X_batch, dists_batch, K_batch, diff_batch, K_batch_exp, dK_batch, dK_dX_batch
            torch.cuda.empty_cache()

        return dK_dX

@register_conditioning_method(name='euclidean')
class EuclideanRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, embed_fn, n_embed, **kwargs)
        self.scale = kwargs.get('scale', 1.0)

    def empirical_denoiser(self, x_t, time, sigma=1.0, **kwargs):
        # shape
        M, C, H, W = self.ref_data.shape
        N, _, _, _ = x_t.shape

        # normalization (for sdv3)
        x_t = x_t / torch.norm(x_t, dim=1, keepdim=True)
        
        # reshape
        x_t_proj = x_t.reshape(-1, C*H*W)
        ref_data_proj = self.ref_data.reshape(-1, C*H*W)
        
        # denominator
        ones = torch.ones(M)[..., None].to(ref_data_proj.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_proj, ones), 1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        # Compute the log of the kernel
        log_kernel = kernel.log()  # Assume kernel values are all positive

        # Compute log of the denominator
        log_denominator = log_kernel[:, -1].reshape(-1, 1)  # Extract the last column in log space
        log_nominator = log_kernel[:, :-1]  # Extract all but the last column in log space
        log_negative_score = log_nominator - log_denominator
        
        # negative_score
        negative_score = log_negative_score.exp()
        negative_score_item = negative_score.mean().item()

        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item


@register_conditioning_method(name='kernel')
class RBFKernelRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, embed_fn, n_embed, **kwargs)
        self.scale = kwargs.get('scale', 1.0)

    def empirical_denoiser(self, x_t, time, sigma=1.0, **kwargs):
        # Embedding for kernel distance
        x_t_proj = self.project(x_t, time)
        ref_data_proj = self.project(self.ref_data, time)

        # normalization (for sdv3)
        x_t = x_t / torch.norm(x_t, dim=1, keepdim=True)
        
        # reshape
        N, D = x_t_proj.shape
        M, D = ref_data_proj.shape
        _, C, H, W = self.ref_data.shape
        x_t_proj = x_t_proj.reshape(1, -1, D)
        ref_data_proj = ref_data_proj.reshape(1, -1, D)
        
        # data augmentation for denominator
        ref_data_org = self.ref_data.reshape(M, -1)
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

@register_conditioning_method(name='kernel_fast')
class RBFKernelRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, ref_data, embed_fn, n_embed, **kwargs)
        self.scale = kwargs.get('scale', 1.0)
        
    def empirical_denoiser(self, x_t, sigma=1.0, **kwargs):
        '''
            args:
                x_t: latent (no need to be embedded)
            returns:
                negative_score: negative score
                negative_score_item: mean of negative score
        '''
        
        # normalization (for sdv3)
        x_t = x_t / torch.norm(x_t, dim=1, keepdim=True)
        
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
    
@register_conditioning_method(name='random_noise')
class RandomNoiseRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, ref_data, embed_fn, n_embed, **kwargs)
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
        
        # Random Noise
        negative_score = torch.randn(size=(1, C*H*W)).to(ref_data_latent.device)
        negative_score_item = negative_score.clamp(min=-1e10, max=1e10).mean().item()
        
        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item    

@register_conditioning_method(name='spell')
class SparseRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, n_embed, **kwargs):
        super().__init__(ref_data, embed_fn, n_embed, **kwargs)
        self.radius = kwargs.get('radius', 1.0)
        self.scale = kwargs.get('scale', 1.0) # overcompensation

    def find_neighbors_within_radius(self, x_0_hat):
        ref_data_latent = self.get_proj_ref() # [N, C, emb_H, emb_W]
        
        if x_0_hat.dim() != ref_data_latent.dim():
            x_0_hat = x_0_hat.unsqueeze(0)
        
        diff = x_0_hat - ref_data_latent
        
        # condition
        repellency_data_idx = torch.norm(diff, p=2, dim=(1,2,3)) < self.radius
        
        # find neighbors
        repellency_data = ref_data_latent[repellency_data_idx]
        return repellency_data
        
    def repellency_force(self, x_0_hat, **kwargs):
        repellency_data = self.find_neighbors_within_radius(x_0_hat)
        diff_vec = x_0_hat.unsqueeze(1) - repellency_data.unsqueeze(0) #[N, M, C, H, W]
        weight = torch.norm(diff_vec, p=2 ,dim=(2,3,4))
        trunc_weight = F.relu((self.radius/weight) - 1.)
        
        # Dimension matching
        trunc_weight = trunc_weight[..., None, None, None]
        repellency_term = (diff_vec * trunc_weight).sum(dim=1)
        return repellency_term, repellency_term.norm(p=2).item()
    
    def empirical_denoiser(self, x_t, **kwargs):
        '''wrapped function'''
        result = self.repellency_force(x_t, **kwargs)
        return result