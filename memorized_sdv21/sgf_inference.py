import yaml
from datetime import datetime
import pandas as pd
from difflib import get_close_matches
import re
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from datasets import insert_rand_word

class Newpipe(StableDiffusionPipeline):
    def _encode_prompt(self,*args, **kwargs):
        embedding = super()._encode_prompt(*args,**kwargs)
        return embedding + self.noiselam * torch.randn_like(embedding)

def resize(w_val,l_val,img):
#   img = Image.open(img)
  img = img.resize((w_val,l_val), Image.Resampling.LANCZOS)
  return img


def prompt_augmentation(prompt, aug_style,tokenizer=None, repeat_num=2):
    if aug_style =='rand_numb_add':
        for i in range(repeat_num):
            randnum  = np.random.choice(100000)
            prompt = insert_rand_word(prompt,str(randnum))
    elif aug_style =='rand_word_add':
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt,randword)
    elif aug_style =='rand_word_repeat':
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt,randword)
    else:
        raise Exception('This style of prompt augmnentation is not written')
    return prompt

def prompt_to_clean_name(prompt: str) -> str:
    s = prompt.strip()
    # Remove variants like "An image of ", "an image of ", "A image of ", etc.
    s = re.sub(r'(?i)^\s*(an?|the)\s+image\s+of\s+', '', s).strip()
    # If the remaining part is like "a garbage truck", remove the leading article
    s = re.sub(r'(?i)^\s*(an?)\s+', '', s).strip()
    # Normalize whitespace + lowercase
    s = re.sub(r'\s+', ' ', s).lower()
    return s

def lookup_wnid(df: pd.DataFrame, class_text: str) -> dict:
    key = class_text.strip().lower()

     # 1) Exact match on clean_name
    m = df[df["clean_name"] == key]
    if not m.empty:
        return m.iloc[0].to_dict()  # {"wnid": "...", "name": "...", "clean_name": "..."}

    # 2) Substring match within name (all aliases), respecting comma boundaries
    alias_match = df[df["name"].str.lower().str.contains(
        rf'(^|,\s*){re.escape(key)}(\s*$|,)', regex=True
    )]
    if not alias_match.empty:
        return alias_match.iloc[0].to_dict()

    # 3) Fuzzy match (slightly different spelling)
    cand = get_close_matches(key, df["clean_name"].tolist(), n=1, cutoff=0.7)
    if cand:
        m = df[df["clean_name"] == cand[0]]
        if not m.empty:
            return m.iloc[0].to_dict()

    raise KeyError(f"Could not find class '{class_text}' in classnames.")

import torch
import argparse
import os
import numpy as np
import json
import ast
from transformers import AutoTokenizer
import glob 
import itertools
from PIL import Image

def main(args):
    if args.modelpath is None:
        savepath = f'./inferences/defaultsd/{args.dataset}/{args.capstyle}'
    else:
        mp = os.path.basename(os.path.normpath(args.modelpath))

        if "traintext" not in args.modelpath:
            if "imagenette" in args.modelpath:
                args.dataset = 'imagenette10'
                savepath = f'./inferences/imagenette10_frozentext/{mp}'
            elif "aesthetics" in args.modelpath:
                args.dataset = 'laionaesthetics'
                savepath = f'./inferences/laionaesthetics_ft/{mp}'
            elif "laion" in args.modelpath:
                args.dataset = 'laion'
                savepath = f'./inferences/laion_frozentext/{mp}'
            elif "l100kaion" in args.modelpath:
                args.dataset = 'l100kaion'
                savepath = f'./inferences/l100kaion_frozentext/{mp}'
            else:
                raise 'Savepath doesnt exist for this case'
        else:
            if "imagenette" in args.modelpath:
                args.dataset = 'imagenette10'
                savepath = f'./inferences/imagenette10_traintext/{mp}'
            elif "laion" in args.modelpath:
                args.dataset = 'laion'
                savepath = f'./inferences/laion_traintext/{mp}'
            else:
                raise 'Savepath doesnt exist for this case'
            
    # savepath
    savepath = os.path.join(savepath, args.nickname)
    
    if args.iternum is not None:
        savepath = f'{savepath}_{args.iternum}'
    # caption style
    savepath = f'{savepath}/{args.modelstyle}'
    if args.rand_noise_lam is not None:
        savepath = f'{savepath}_ginfer{args.rand_noise_lam}'
    if args.rand_augs is not None:
        savepath = f'{savepath}_auginfer_{args.rand_augs}_{args.rand_aug_repeats}'
    os.makedirs(savepath,exist_ok=True)
    os.makedirs(f'{savepath}/generations',exist_ok=True)
    if args.modelpath is None:
        checkpath = "stabilityai/stable-diffusion-2-1" 
    elif args.iternum is not None:
        checkpath = f'{args.modelpath}/checkpoint_{str(args.iternum)}/'   
    else:
        checkpath = f'{args.modelpath}/checkpoint/'
    if args.modelpath is None:
        device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained(checkpath, use_auth_token=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = lambda images, clip_input: (images, False)
        pipe = pipe.to(device)
        generator = torch.Generator(device=device).manual_seed(42)
    elif args.rand_noise_lam is not None: 
        # raise 'Code not written yet, TODO!'
        pipe = Newpipe.from_pretrained(
            checkpath,safety_checker=None
        ).to("cuda")
        pipe.noiselam = args.rand_noise_lam
    else:
        from models.sgf_pipeline import SGFPipeline
        pipe = SGFPipeline.from_pretrained(
            checkpath, safety_checker=None
        ).to("cuda")
        gen = torch.Generator(device="cuda")


    tokenizer = AutoTokenizer.from_pretrained(
            checkpath,
            subfolder="tokenizer",
            # revision=args.revision,
            use_fast=False,
        )
    # generator = torch.Generator("cuda").manual_seed(42)

    num = args.im_batch
    num_batches = args.nbatches
    count = 0
    prompt_list = None
    
    if args.modelstyle == "nolevel":
        prompt_list = ["An image"]*num_batches
    elif args.modelstyle == "classlevel":
        objects = [
            'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute' 
        ]
        '''random 
        np.random.seed(args.seed)
        temp = list(np.random.choice(objects, num_batches))
        prompt_list = [f"An image of {x}" for x in temp]
        '''
        m = len(objects)
        q, r = divmod(num_batches, m)    # First r objects get q+1, the rest get q

         # Repeat counts for each object (max deviation is 1)
        counts = [q + (1 if i < r else 0) for i in range(m)]

        # Concatenate object names in blocks (keep order, no shuffle)
        temp = [obj for i, obj in enumerate(objects) for _ in range(counts[i])]

        # Prompts
        prompt_list = [f"An image of {x}" for x in temp]

        # Seed assignment: within each object block, assign 0..(block_len-1) sequentially
        #  - If block length is q:   0..q-1
        #  - If block length is q+1: 0..q   (i.e., from 0 up to num_batches/10)
        from itertools import chain
        seeds = list(chain.from_iterable(range(c) for c in counts))

        # Sanity check (length match)
        assert len(prompt_list) == len(seeds) == num_batches
        
    
    elif args.modelstyle in ["instancelevel_blip","instancelevel_random"]:
        if args.dataset == 'imagenette10':
            if args.modelstyle == 'instancelevel_blip': 
                prompt_json = './data/imagenette2-320/blip_captions.json'
            else:
                prompt_json = './data/imagenette2-320/random_captions_4.json'
        elif args.dataset == 'laionaesthetics':
            if args.modelstyle == 'instancelevel_blip':
                prompt_json = './data/laion_10k_random_aesthetics_5plus/laion_aesthetics_combined_captions.json'
            else:
                raise Exception('Case not written')
        elif args.dataset == 'laion':
            if args.modelstyle == 'instancelevel_blip':
                prompt_json = './data/laion_10k_random/laion_combined_captions.json'
            else:
                raise Exception('Case not written')
        elif args.dataset == 'l100kaion':
            if args.modelstyle == 'instancelevel_blip':
                prompt_json = './data/laion_100k_random_sdv2p1/l100kaion_combined_captions.json'
            else:
                raise Exception('Case not written')

        with open(prompt_json) as f:
            all_prompts_dict = json.load(f)
        
    
        okprompts = [v[0] for k,v in all_prompts_dict.items()]
        if args.dataset == 'l100kaion':
            okprompts = okprompts[:10000]
        np.random.seed(args.seed)
        prompt_list = list(np.random.choice(okprompts,num_batches))

        if args.modelstyle == "instancelevel_random":
            new_prompts = []
            for p in prompt_list:
                instance_prompt = ast.literal_eval(p)
                instance_prompt = tokenizer.decode(instance_prompt)
                new_prompts.append(instance_prompt)
            prompt_list = new_prompts[:]
    if args.rand_augs is not None:
        final_prompt_list = []
        for prompt in prompt_list:
            newprompt = prompt_augmentation(prompt, args.rand_augs, tokenizer, args.rand_aug_repeats)
            final_prompt_list.append(newprompt)
        prompt_list = final_prompt_list

    # save the prompt list
    with open(f'{savepath}/prompts.txt', 'w') as f:
        for line in prompt_list:
            f.write(f"{line}\n")

    ##############
    # REPELLENCY #
    ##############
    from utils.sgf_utils import load_yaml
    from utils.sgf_data import get_dataset, get_dataloader, get_transform, load_imagenet_classnames, get_class_subset_loader, get_all_imgs_imageFolder
    from repellency.repellency_methods_sgf import get_repellency_method
    
    if args.task_config is not None:
        task_config = load_yaml(args.task_config)

        # save config
        with open(f'{savepath}/cfg.yaml', "w", encoding="utf-8") as f:
            yaml.safe_dump(task_config,f,sort_keys=False, allow_unicode=True, default_flow_style=False)

        # Prepare dataloader
        data_config = task_config['data']

        # text prompts for classname
        classnames_df = load_imagenet_classnames(data_config['root'])

        transform = get_transform(**data_config)
        repellency_dataset = get_dataset(**data_config, transforms=transform)
        repellency_loader = get_dataloader(repellency_dataset, batch_size=200, num_workers=0, train=False)
        
        # embed_fn
        embed_fn = lambda x : pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
        repellency_config = task_config['repellency']

        # extract time-steps for neg
        neg_config = {
            "neg_start": repellency_config.pop("neg_start"),
            "neg_end": repellency_config.pop("neg_end")}

        print(f"Repellency method : {task_config['repellency']['method']}")
    ################################################
    
    # Denoising
    for i in range(num_batches):
        if prompt_list is not None:
            prompt = prompt_list[i]
            seed = seeds[i]
        else:
            raise "no prompt list!"
        
        # Repellency
        if i < num_batches-2:
            if i == 0 or prompt_list[i] != prompt_list[i+1]:
                # extract wnid
                cls_text = prompt_to_clean_name(prompt)
                data = lookup_wnid(classnames_df, cls_text)
                
                repellency_loader = get_class_subset_loader(repellency_dataset, target_class=data["wnid"], batch_size=200, num_workers=0, shuffle=False)
                imgs_by_class = get_all_imgs_imageFolder(repellency_loader)
                ref_imgs = imgs_by_class.to('cuda')
                repellency_processor = get_repellency_method(repellency_config['method'], 
                                                        ref_data = ref_imgs,
                                                        embed_fn=embed_fn,
                                                        n_embed = repellency_config['n_embed'],
                                                        scheduler = pipe.scheduler,
                                                        **repellency_config['params'])
            
        if args.modelpath is None:
            images = pipe(prompt, num_inference_steps=50, generator=generator).images
        else:
            images = pipe(prompt=prompt, 
                          height=args.resolution, 
                          width=args.resolution,
                          num_inference_steps=50, 
                          generator=gen.manual_seed(seed),
                          num_images_per_prompt=args.im_batch,
                          repellency_processor=repellency_processor,
                          # budget for negative guidance
                          **neg_config,
                        ).images
        
        for j in range(len(images)):
            image = images[j]
            if image.size[0] > args.resolution:
                image = resize(args.resolution, args.resolution, image)
            # import ipdb; ipdb.set_trace()
            image.save(f"{savepath}/generations/{count}.png")
            count+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument("--modelpath", type=str, default=None)#required=True)
    parser.add_argument("--dataset", type=str, required=None)
    # parser.add_argument("--modelstyle", type=str, required=True)
    parser.add_argument("--capstyle", type=str, default=None)
    parser.add_argument("--captoken", type=str, default=None)

    # nickname for savepath
    parser.add_argument('--nickname', type=str, default="", help="run nickname (leave empty to auto-set to the current timestamp)")

    # SGF
    parser.add_argument('--task_config', type=str, default="configs_sgf/sgf.yaml", help="Path to the task configuration file.")

    # parser.add_argument('--synset_map', type=str, default=None)
    parser.add_argument("-nb","--nbatches", type=int, required=True)
    parser.add_argument("-imb","--im_batch", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--iternum", default=None, type=int)
    parser.add_argument("--rand_noise_lam", type=float, default=None)
    parser.add_argument("--rand_augs", type=str, default=None)
    parser.add_argument("--rand_aug_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")


    args = parser.parse_args()

    # 닉네임 없으면 현재 시각으로 대체
    if not args.nickname.strip():
        args.nickname = datetime.now().strftime("%Y%m%d%H%M%S")

    assert not (args.modelpath is None and args.capstyle is None), "Modelpath and caption style cant be None at the same time"
    assert not (args.modelpath is None and args.dataset is None), "Modelpath and Dataset name cant be None at the same time"

    
    if args.modelpath is None:
        print('Default SD generations will be done')
    
    if args.capstyle is not None and args.capstyle in ['nolevel','classlevel','instancelevel_blip','instancelevel_random']:
        args.modelstyle = args.capstyle
    elif 'nolevel' in args.modelpath:
        args.modelstyle = 'nolevel'
    elif 'classlevel' in args.modelpath:
        args.modelstyle = 'classlevel'
    elif 'instancelevel_blip' in args.modelpath:
        args.modelstyle = 'instancelevel_blip'
    elif 'instancelevel_random' in args.modelpath:
        args.modelstyle = 'instancelevel_random'

    if args.rand_augs:
        assert args.modelstyle == "instancelevel_blip", "Random caption augmentations can only be applied if model is trained on blip captions"
    main(args)
