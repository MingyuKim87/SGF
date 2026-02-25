import os
from glob import glob
import random
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torchvision import transforms
from diffusers import AudioDiffusionPipeline
from torchvision.transforms.functional import to_tensor

IMAGENET_PATH="datasets/imagenet/train"
CELEBA_PATH="datasets/celeba_hq_256"

def load_image_dataset(dataset, num_samples=-1, target=-1, return_tensor=True, normalize=True, dataset_dir=None):
    if dataset == 'sample':
        images = []
        if num_samples < 0:
            num_samples = len(os.listdir(dataset_dir))
        for img in os.listdir(dataset_dir)[:num_samples]:
            
            images.append(Image.open(os.path.join(dataset_dir, img)).convert('RGB'))        
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
    elif dataset == 'sample_dict':
        images = []
        images_name = []
        if num_samples < 0:
            num_samples = len(os.listdir(dataset_dir))
        
        for img in os.listdir(dataset_dir)[:num_samples]:
            images_name.append(img)
            images.append(Image.open(os.path.join(dataset_dir, img)).convert('RGB'))        
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    

    elif dataset == 'cat':
        images = load_dataset("cats_vs_dogs")
        images = images.filter(lambda x: x == 0, input_columns='labels')
        images = images['train'][:num_samples]['image']
        if not return_tensor:
            images = [images.resize((256, 256)) for images in images]
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    
    elif dataset == 'cifar10':
        dataset = load_dataset('cifar10')
        
        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('img', 'images')
        dataset = concatenate_datasets([dataset['train'], dataset['test']])
        
        if num_samples > 0:
            dataset = dataset[:num_samples]

        images = [images.resize((32, 32)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    elif dataset == 'imagenet':
        dataset = load_from_disk(IMAGENET_PATH)

        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('image', 'images')
        
        if num_samples > 0:
            dataset = dataset[:num_samples]
        
        images = [images.resize((256, 256)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif dataset == "celeba":
        images = []
        fpaths = sorted(glob(f'{dataset_dir}/*.jpg', recursive=True))
        
        if num_samples < 0:
            num_samples = len(fpaths)
        for img in fpaths[:num_samples]:
            images.append(Image.open(img).convert('RGB'))
        tf = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
        
    elif dataset == 'celebahq':
        images = []
        
        if num_samples < 0:
            num_samples = len(os.listdir(CELEBA_PATH))
        
        for img in os.listdir(CELEBA_PATH)[:num_samples]:
            images.append(Image.open(os.path.join(CELEBA_PATH, img)))
        
        tf = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

    elif dataset == 'bird-species':
        
        dataset = load_dataset('chriamue/bird-species-dataset')
        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
        
        if target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')
        
        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('image', 'images')
        
        if num_samples > 0:
            dataset = dataset[:num_samples]
        
        images = [images.resize((256, 256)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        raise NotImplementedError

    if normalize:
        tf.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if return_tensor:
        image_tensors = [tf(img) for img in images]
        return torch.stack(image_tensors, dim=0)
    else:
        if dataset == 'sample_dict':
            return {'images': images, 
                    'images_name' : images_name}
        
        return images