import os
import torch
from sdv3_pipeline import StableDiffusion3Pipeline
import argparse
from datasets import load_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')


    # parser.add_argument('--dataset_name',type=str, default='p4d')
    # parser.add_argument('--dataset_name',type=str, default='ring-a-bell')
    # parser.add_argument('--dataset_name',type=str, default='mma')
    parser.add_argument('--dataset_name',type=str, default='unleardiff')
    # parser.add_argument('--dataset_name',type=str, default='coco')

    # parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/P4D/data/p4d/p4dn_16_prompt.csv')
    # parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/references/RECE/dataset/nudity-ring-a-bell.csv')
    # parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/references/MMA-Diffusion-NSFW-adv-prompts-benchmark/mma-diffusion-nsfw-adv-prompts.csv')
    parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/references/RECE/dataset/nudity.csv')
    # parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/references/RECE/dataset/coco_30k.csv')
    # parser.add_argument('--csv', type=str, default='/nas-ssd2/jhyoon/safe_vidgen/references/RECE/dataset/coco_30k_1000.csv')
    
    parser.add_argument('--save_path', type=str, default='/nas-hdd/shoubin/result/safegen/sdv3_ours_unlearn')
    # parser.add_argument('--save_path', type=str, default='/nas-hdd/shoubin/result/safegen/sdv3_coco')
    
    args = parser.parse_args()


    device = 'cuda:0'
    
    pipe = StableDiffusion3Pipeline.from_pretrained("/nas-hdd/shoubin/pretrained_model/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    dataset = load_dataset('csv',data_files=args.csv)
    output_folder = f"{args.save_path}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.dataset_name == 'ring-a-bell':
        key = 'sensitive prompt'
        have_seed = False

    elif args.dataset_name == 'mma':
        key = 'adv_prompt'
        have_seed=False

    elif args.dataset_name == 'unleardiff' or args.dataset_name =='coco' or args.dataset_name =='p4d':
        key = 'prompt'
        seed_key = 'evaluation_seed'
        have_seed=True

    for n, prompt in enumerate(dataset['train'][key]):
        # prompt = args.prompt
        
        if have_seed:
            seed = int([_ for _ in dataset['train'][seed_key]][n])
            generator = torch.Generator(device="cpu").manual_seed(seed)
            # images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images, generator=generator).images
            image = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
                generator=generator
            ).images[0]
        
        
        else:
            seed = 42
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
                generator=generator
            ).images[0]

        # for i, im in enumerate(images):
        image.save(f"{output_folder}/{n}.jpg")  
        # break