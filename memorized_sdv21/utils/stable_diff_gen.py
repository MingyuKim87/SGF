## huggingface-cli login
# https://github.com/huggingface/diffusers/blob/7c2262640bbf9fa61c281bc49eb8494cb48da81f/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py


## to run python stable_diff_gen.py -tpf /fs/cml-projects/diffusion_rep/data/sdprompts/poloclub/sdprompts100k_chunk_0.csv -sp /fs/cml-projects/diffusion_rep/data/sdprompts/generation_poloclub_100k


import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import requests
import io
import pandas as pd
import argparse
import os
import numpy as np
from pathlib import Path



def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def resize(w_val,l_val,img):
#   img = Image.open(img)
  img = img.resize((w_val,l_val), Image.Resampling.LANCZOS)
  return img

def chunker(seq, size):
       for pos in range(0, len(seq), size):
           yield seq.iloc[pos:pos + size] 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image guided diffusion')
    parser.add_argument('-im','--image_path', type=str, default=None)
    parser.add_argument('-tpf','--text_prompt_file', type=str, required=True)
 

    parser.add_argument('-sp','--image_save_path', type=str, default='./sdgens_nov5')

    # parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    parser.add_argument('-bs','--batch_size', default=1, type=int)
    parser.add_argument('-nb','--n_batches', default=1, type=int)
    parser.add_argument('-gs','--guidance_scale', default=7.5, type=float)
    parser.add_argument('-sgth','--strength', default=0.5, type=float)
    # parser.add_argument('-lol','--locorall',default='-1', type=str)
    parser.add_argument('--set_seed', default=42, type=int)
    parser.add_argument('--extraname', type=str)
    parser.add_argument('--chunk', default=0, type=int)


    args = parser.parse_args()

    prompt_file = pd.read_csv(args.text_prompt_file)

    # for i in chunker(prompt_file, 4):
    #    print(i)

    model_id = "stabilityai/stable-diffusion-2-1"
    device = "cuda"

    os.makedirs(args.image_save_path,exist_ok=True)
    # import ipdb; ipdb.set_trace()
    steps = 50

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(device)
    
    # for index, row in prompt_file.iterrows():
        # print(index, row['prompt'])
    count = 0
    for dfchunk in chunker(prompt_file, 1):

        prompt_list = dfchunk['prompt'].tolist()
        print(prompt_list)
        
        # with autocast("cuda"):
        num = args.batch_size
        generator = torch.Generator(device=device).manual_seed(args.set_seed)
        my_file = Path(f"{args.image_save_path}/gen_chunk{args.chunk}_{count}.png")
        if not my_file.is_file() and prompt_list != [np.nan]:
            images = pipe(prompt_list, num_inference_steps=50, generator=generator).images

            # for j in range(num):
            # image = resize(224, 224, images[j])
            image = resize(224, 224, images[0])
            image.save(f"{args.image_save_path}/gen_chunk{args.chunk}_{count}.png")
        else:
            print(f"Skipped:{count}")
        count+=1

