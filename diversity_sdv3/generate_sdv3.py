import numpy as np
# from torchvision import transforms
import pandas as pd
import argparse
import torch
# import csv
import os
import json
# from einops import rearrange

from PIL import Image
import albumentations as A

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler

from main_utils import Logger, read_json, dummy, horz_stack, vert_stack, load_yaml, save_combined_config

# models
from models.sdv3.sdv3_pipeline import StableDiffusion3Pipeline
# from models.sdv3.safree_pipeline import StableDiffusion3Pipeline as SafreeDiffusion3Pipeline
from models.sdv3.safree_pipeline_efficient import StableDiffusion3Pipeline as SafreeDiffusion3Pipeline
from models.sdv3.safer_pipeline import StableDiffusion3Pipeline as SaferDiffusion3Pipeline
from models.sdv3.safer_pipeline_no_early_stop import StableDiffusion3Pipeline as SpellDiffusion3Pipeline

# clip
import pickle
import clip
import open_clip
from transformers import CLIPProcessor, CLIPModel # clip

# evaluation
from evaluations.fid import evaluate_fid

# huggingface dataset : coco-30k
from datasets import load_dataset

# Repellency
from data.dataloader import get_dataset, get_dataloader, get_transform, get_all_imgs, get_all_imgs_by_target_class, get_all_imgs_imageFolder, load_imagenet_classnames, get_class_subset_loader
from repellency.repellency_methods_fast_sdv3 import get_repellency_method

SD_FUNCTIONS = {
    "std" : StableDiffusion3Pipeline,
    "sgf": SaferDiffusion3Pipeline, # main table t \in \{1000, 800\}
}

SLD_CONFIGS = {
    "MAX": SafetyConfig.MAX,
    "STRONG":  SafetyConfig.STRONG,
    "MEDIUM": SafetyConfig.MEDIUM,
    "WEAK": SafetyConfig.WEAK
}

def merge_dicts_append(dict1, dict2):    
    for key in dict2:
        assert key in dict1.keys()        
        if isinstance(dict2[key], list):
            dict1[key].append(dict2[key])
        else:
            dict1[key] += [dict2[key]]        
    return dict1


'''
def load_dataset(data_f, category, rm_cols, debug=False):
    # Read CSV file
    dataset = pd.read_csv(data_f)
    # delete index column
    
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])  
    dataset["nudity_toxicity"] = 0.0
    
    print(f"{category} dataset size: {dataset.shape[0]}")
    return dataset
'''


def load_sd(pipeline_func, device, weight_dtype, unet_ckpt=None):
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    # scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    # scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    pipe = pipeline_func.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
        )
    
    if unet_ckpt is not None:
        unet_weight = torch.load(unet_ckpt, map_location='cpu')
        try:
            pipe.unet.load_state_dict(unet_weight)
        except:
            pipe.unet.load_state_dict(unet_weight['unet'])
        print(f"ESD unet: {unet_ckpt} is loaded...")
        
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    gen = torch.Generator(device=device)
    return pipe, gen

class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name, 
                                                        device, 
                                                        jit=False)
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, 
                                                                    keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_norm @ embeddings_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()
    
    
# def initialize_prompts(clip_model, text_prompts, device):
#     text = clip.tokenize(text_prompts).to(device)
#     return clip_model.encode_text(text)


# def save_prompts(classifier, save_path):
#     prompts = classifier.embeddings.detach().cpu().numpy()
#     pickle.dump(prompts, open(save_path, 'wb'))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, 'rb'))).to(device)

class Eval:
    def __init__(self, args):
        self.category = args.category
        if self.category == 'nudity':
            self.nude = Classifier(args.nudenet_path)
            print(f"NudeNet: {args.nudenet_path} is loaded...")
        elif 'artists-' in self.category:
            pass
        elif 'coco_open_clip' in self.category:
            # open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=args.device)
            self.device = args.device
            print(f"CLIP model {args.clip_model}: {args.clip_pretrain} is loaded...")
        elif 'coco' in self.category:
            self.device = args.device
            # model_name = "openai/clip-vit-base-patch32"
            model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_preprocess = CLIPProcessor.from_pretrained(model_name)
        elif self.category == 'all':
            ValueError("Currently, only nudity or artist category are supported.")

    def compute_embeddings(self, images):
        pr_imgs = [self.clipw.preprocess(img) for img in images]
        pr_imgs = torch.stack(pr_imgs).to('cuda')
        return self.clipw(pr_imgs).half()
    
    def measure_similarity(self, orig_images, images):
        ref_model = self.clip_model
        ref_clip_preprocess = self.clip_preprocess
        device = self.device
        
        with torch.no_grad():
            ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
            if torch.__version__ >= "1.13.0":
                ori_batch = torch.concatenate(ori_batch).to(device)
            else:
                ori_batch = torch.cat(ori_batch).to(device)

            gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
            if torch.__version__ >= "1.13.0":
                gen_batch = torch.concatenate(gen_batch).to(device)
            else:
                gen_batch = torch.cat(gen_batch).to(device)
            
            ori_feat = ref_model.encode_image(ori_batch)
            gen_feat = ref_model.encode_image(gen_batch)
            
            ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
            gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
            
            return (ori_feat @ gen_feat.t()).mean().item()
        
    def clip_score(self, images, prompts):
        ref_model = self.clip_model
        ref_clip_preprocess = self.clip_preprocess
        device = self.device
        
        inputs = ref_clip_preprocess(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # inference
        with torch.no_grad():
            image_features = ref_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = ref_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        similarity = torch.matmul(image_features, text_features.T).mean().item()
        return similarity
        
    def __call__(self, samples, threshold=0.6, org_samples=None, prompts=None):        
        is_nude = False
        if self.category == 'nudity':
            img_names = [f"{i}.png" for i in range(len(samples))]
            # import pdb; pdb.set_trace()
            preds = self.nude.classify(images=samples, image_names=img_names)
            res_nude = [v['unsafe'] for k, v in preds.items() if 'unsafe' in v.keys()]
            is_nude = any(prob>=threshold for prob in res_nude)
            pred = max(res_nude)
            unsafe = bool(is_nude)
        elif self.category == 'coco_open_clip':
            # pred == similarity_metric
            pred = similarity_metric = self.measure_similarity(images=samples, orig_images=org_samples)
            unsafe = None
        elif self.category == "coco":
            # pred == clip_score
            pred = clip_score = self.clip_score(images=samples, 
                                                prompts=prompts)
            unsafe = None
        elif self.category == 'all':
            embs = self.compute_embeddings(samples)
            preds = self.q16_classifier(embs)
            is_nude = torch.argmax(preds, dim=-1) # label 1 corrosponds to inappropriate material
            pred = torch.max(preds, dim=-1)
            if len(samples) == 1:
                unsafe = bool(is_nude.detach().cpu().data)
            else:
                unsafe = bool(max(is_nude.detach().cpu().data))
        return unsafe, pred
        
def main():
    # output save paths
    safe_imgdir = os.path.join(args.save_dir, "safe")
    unsafe_imgdir = os.path.join(args.save_dir, "unsafe")
    all_imgdir = os.path.join(args.save_dir, "all")
    ref_imgdir = os.path.join(args.save_dir, "ref")

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        os.mkdir(all_imgdir)
        
    
    task_config = None
    log_f = os.path.join(args.save_dir, "logs.txt")
    logger = Logger(log_f)
    logger.log("All configurations provided:")
    for arg in vars(args):
        logger.log(f"{arg}: {getattr(args, arg)}")

    # load a model
    erase_id = args.erase_id if 'xl' not in args.model_id else args.erase_id + '+xl'
    logger.log(f"Erase_path: {args.erase_concept_checkpoint if not 'std' in args.erase_id else 'na'}")
    pipe, gen = load_sd(SD_FUNCTIONS[erase_id], args.device, torch.float32, args.erase_concept_checkpoint)

    ##############
    # REPELLENCY #
    ##############
    if args.task_config is not None:
        # mean processor
        task_config = load_yaml(args.task_config)
            
        # Prepare dataloader
        data_config = task_config['data']

        # text prompts for classname
        dataset = load_imagenet_classnames(data_config['root'])

        transform = get_transform(**data_config)
        repellency_dataset = get_dataset(**data_config, transforms=transform)
        repellency_loader = get_dataloader(repellency_dataset, batch_size=200, num_workers=0, train=False)
        
        # embed_fn
        embed_fn = lambda x : pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
        repellency_config = task_config['repellency']
        logger.log(f"Repellency method : {task_config['repellency']['method']}")
    ################################################
        
 
    if args.category == "nudity" or args.category == "coco":    
        ####################
        # NEGATIVE SPACES  #
        ####################
        if not "safree" in args.erase_id:
            negative_prompt_space =  [" "]
        else:
            # For safree
            negative_prompt_space = [
                "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
                "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
                "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
                "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
            ]
    elif "artists-" in args.category:
        negative_prompt_space = args.category.split("-")[-1]
        negative_prompt_space = ["Van Gogh"] if negative_prompt_space == "VanGogh" else negative_prompt_space
        negative_prompt_space = ["Kelly McKernan"] if negative_prompt_space == "KellyMcKernan" else negative_prompt_space
        
    ###################
    # NEGATIVE PROMPT #
    ###################
    negative_prompt = None
    
    vstart, vend = args.valid_case_numbers.split(',')
    dataset = dataset[int(vstart):]
    dataset = dataset[:int(vend)]
    
    for _iter, data in (dataset.iterrows() if isinstance(dataset, pd.DataFrame) else enumerate(dataset)):
        # create class dir
        class_dir = os.path.join(ref_imgdir, data["wnid"]); os.makedirs(class_dir, exist_ok=True)
        
        target_prompt = "a photo of " + data["name"]

        # Repellency
        repellency_loader = get_class_subset_loader(repellency_dataset, target_class=data["wnid"], batch_size=200, num_workers=0, shuffle=False)
        imgs_by_class = get_all_imgs_imageFolder(repellency_loader)
        ref_imgs = imgs_by_class
        repellency_processor = get_repellency_method(repellency_config['method'], 
                                                ref_data = ref_imgs,
                                                embed_fn=embed_fn,
                                                n_embed = repellency_config['n_embed'],
                                                scheduler = pipe.scheduler,
                                                **repellency_config['params'])
        
        # extract time-steps for neg
        neg_config = {
            "neg_start": repellency_config.pop("neg_start"),
            "neg_end": repellency_config.pop("neg_end")}

        
        # Guidance
        if hasattr(args, 'guidance_scale'):
            guidance = args.guidance_scale
        else:
            guidance = 7.5
        
        # check if data is broken
        if not isinstance(target_prompt, str) or not isinstance(guidance, (int, float)):
            continue
        
        seeds = list(range(0, 50, 1))
        for j, seed in enumerate(seeds):
            logger.log(f"Seed: {seed}, Class Idx: {_iter}, Case#: {j}: target prompt: {target_prompt}")
        
            # SDv3 Pipeline
            imgs = pipe(
                    target_prompt,
                    negative_prompt="",
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=guidance,
                    generator=gen.manual_seed(seed),
                    height=args.image_length,
                    width=args.image_length,
                    repellency_processor=repellency_processor if args.task_config is not None else None,
                    **neg_config
            ).images
        
            # stack and save the output images
            _save_path = os.path.join(class_dir, f"{j:02d}.png")
            imgs[0].save(_save_path)
        
    # Save Config
    save_combined_config(args, os.path.join(args.save_dir, "config.yaml"), task_config) # config
    
            
    

    '''
    ####################
    # EVALUATION : FID #
    ####################
    evaluate_fid(sample_dir=all_imgdir,
             dataset_root=org_imgdir,
             batch_size=4,
             device=args.device)
    '''

    print('end')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ###################
    # # ARGS FOR CFGS #
    ###################
    parser.add_argument("--config", default="sample_config.json", type=str, help="config file path")
    args, unknown = parser.parse_known_args()  # Config 파일 경로만 먼저 읽기
    cfg = read_json(args.config)  # JSON 로드
    
    ###############
    # # MAIN ARGS #
    ###############
    parser.add_argument("--data", type=str, default=cfg.get("data", "./data/tmp_prompt.csv"))
    parser.add_argument("--save-dir", type=str, default=cfg.get("save_dir", "./results/tmp"))
    parser.add_argument("--erase_id", type=str, default=cfg.get("erase_id", "std"))
    parser.add_argument("--model_id", type=str, default=cfg.get("model_id", "CompVis/stable-diffusion-v1-4"))
    parser.add_argument("--num-samples", type=int, default=cfg.get("num_samples", 1), help="number of images to generate with SD")
    parser.add_argument("--nudenet-path", type=str, default=cfg.get("nudenet_path", "./pretrained/nudenet_classifier_model.onnx"), help="nudenet classifer checkpoint path")
    parser.add_argument("--category", type=str, default=cfg.get("category", "coco"), choices=['coco', 'coco_open_clip'])
    parser.add_argument("--device", default=cfg.get("device", "cuda:0"), type=str, help="first gpu device")
    parser.add_argument("--nudity_thr", default=cfg.get("nudity_thr", 0.6), type=float)
    parser.add_argument("--valid_case_numbers", default=cfg.get("valid_case_numbers", "0,100000"), type=str)
    parser.add_argument("--erase_concept_checkpoint", default=cfg.get("erase_concept_checkpoint", None), type=str)
    
    # base args
    parser.add_argument("--prompt_len", type=int, default=cfg.get("prompt_len", 16))
    parser.add_argument("--every_k", type=int, default=cfg.get("every_k", 3))
    parser.add_argument("--max_length", type=int, default=cfg.get("max_length", 77))
    parser.add_argument("--iter", type=int, default=cfg.get("iter", 3000))
    parser.add_argument("--eval_step", type=int, default=cfg.get("eval_step", 50))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", None))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 0.1))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 0.1))
    parser.add_argument("--prompt_bs", type=int, default=cfg.get("prompt_bs", 1))
    parser.add_argument("--loss_weight", type=float, default=cfg.get("loss_weight", 1.0))
    parser.add_argument("--print_step", type=int, default=cfg.get("print_step", 100))
    parser.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 1))
    parser.add_argument("--image_length", type=int, default=cfg.get("image_length", 512))
    parser.add_argument("--guidance_scale", type=float, default=cfg.get("guidance_scale", 3.5))
    parser.add_argument("--num_inference_steps", type=int, default=cfg.get("num_inference_steps", 50))
    parser.add_argument("--num_images_per_prompt", type=int, default=cfg.get("num_images_per_prompt", 1))
    parser.add_argument("--q16_path", type=str, default=cfg.get("q16_path", "./pretrained/Q16_prompts.p"))
    parser.add_argument("--clip_model", type=str, default=cfg.get("clip_model", "ViT-H-14"))
    parser.add_argument("--clip_pretrain", type=str, default=cfg.get("clip_pretrain", "laion2b_s32b_b79k"))
    parser.add_argument("--target_prompts", type=str, default=cfg.get("target_prompts", None))
    parser.add_argument("--negative_prompts", type=str, default=cfg.get("negative_prompts", None))
    
    # repellency
    parser.add_argument('--task_config', type=str, default=cfg.get("task_config", None), help="Path to the task configuration file.")
    parser.add_argument('--param', type=str, default=cfg.get("param", None), help="Params.")

    # SLD
    parser.add_argument("--safe_level", type=str, default=cfg.get("safe_level", "WEAK"))
    
    # Safe + Free ? --> SAFREE!
    parser.add_argument("--safree", action="store_true", default=cfg.get("safree", False))
    parser.add_argument("--self_validation_filter", "-svf", action="store_true", default=cfg.get("svf", False))
    parser.add_argument("--latent_re_attention", "-lra", action="store_true", default=cfg.get("lra", False))
    parser.add_argument("--sf_alpha", default=cfg.get("sf_alpha", 0.01), type=float)
    parser.add_argument("--re_attn_t", default=cfg.get("re_attn_t", "-1,1001"), type=str)
    parser.add_argument("--freeu_hyp", default=cfg.get("freeu_hyp", "1.0-1.0-0.9-0.2"), type=str)
    parser.add_argument("--up_t", default=cfg.get("up_t", 10), type=int)
    args = parser.parse_args()

    main()
