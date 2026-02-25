from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Evaluator
from .base_image import ImageEvaluator
from .utils.load_data import load_image_dataset

##################
# FID EVALUATION #
##################
def evaluate_fid(sample_dir="results", 
             dataset="sample", # sample means to customized dataset only containing images 
             dataset_root=None,
            #  split='val',
             batch_size=None, 
             device=None,
             **kwargs):
    '''
        FID Evaluation
        args:
            samples_dir : root_path for generated images
            dataset_root : rooth path of (ref) dataset (e.g., datasets/celeba_hq/)
            split : split name (e.g., val, test)
    '''
    # kwargs
    filename = kwargs.get('filename', 'metrics')
    
    # root dir
    root_dir = os.path.dirname(sample_dir)
    
    # samples
    samples = load_image_dataset(dataset=dataset, 
                                 dataset_dir=sample_dir,
                                 return_tensor=False)
    
    # ref images
    ref_images = load_image_dataset(dataset=dataset, 
                                    dataset_dir=dataset_root,
                                    return_tensor=False)
    

    # evaluator
    evaluator = ImageEvaluator(dataset=dataset, 
                               dataset_root=dataset_root,
                               batch_size=batch_size, 
                               device=device)
    
    # container
    metrics = {}
    
    # evaluate
    kid = evaluator._compute_kid(samples, ref_images)
    fid = evaluator._compute_fid(samples, ref_images)
    
    metrics['fid'] = float(fid)
    metrics['kid'] = float(kid)
    metrics['log_kid'] = float(np.log(kid))
    
    # save results
    yaml.dump(metrics, open(os.path.join(root_dir, f'{filename}.yaml'), 'w'))
    
    return metrics


###############
# CLIP SCORES #
###############
def evaluate_clip_score(sample_dir="results", 
             dataset="sample", # sample means to customized dataset only containing images 
             prompts_csv=None,
            #  split='val',
             batch_size=None, 
             device=None,
             **kwargs):
    '''
        FID Evaluation
        args:
            samples_dir : root_path for generated images
            prompts : text prompts (e.g., ['a photo of a cat', 'a photo of a dog'])
    '''
    # kwargs
    filename = kwargs.get('filename', 'metrics')
    
    # root dir
    root_dir = os.path.dirname(sample_dir)
    
    # samples
    samples = load_image_dataset(dataset='sample_dict', 
                                 dataset_dir=sample_dir,
                                 return_tensor=False)
    
    # text prompts
    img_names = samples['images_name']
    samples = samples['images']

    # extract file names
    img_ids = [int(name.replace('.png', '')) for name in img_names]

    # finding caption in prompts_csv corresponding to image_id
    matched_prompts = prompts_csv.set_index('image_id').loc[img_ids, 'caption'].tolist()
    
    # evaluator
    evaluator = ImageEvaluator(dataset=dataset, 
                               batch_size=batch_size, 
                               device=device)
    
    # container
    metrics = {}
    
    # evaluate
    clip_score = evaluator._compute_clip_score(samples, matched_prompts)
    metrics['clip_score'] = float(clip_score)
    
    # save results
    yaml.dump(metrics, open(os.path.join(root_dir, f'{filename}.yaml'), 'w'))
    
    return metrics
    
def evaluate_clip_score_CoPro(sample_dir="results", 
             dataset="sample", # sample means to customized dataset only containing images 
             prompts_csv=None,
            #  split='val',
             batch_size=None, 
             device=None,
             **kwargs):
    '''
        FID Evaluation
        args:
            samples_dir : root_path for generated images
            prompts : text prompts (e.g., ['a photo of a cat', 'a photo of a dog'])
    '''
    # kwargs
    filename = kwargs.get('filename', 'metrics')
    
    # root dir
    root_dir = os.path.dirname(sample_dir)
    
    # samples
    samples = load_image_dataset(dataset='sample_dict', 
                                 dataset_dir=sample_dir,
                                 return_tensor=False)
    
    # text prompts
    img_names = samples['images_name']
    samples = samples['images']

    # extract file names
    # img_ids = [int(name.replace('.png', '')) for name in img_names]
    img_ids = [int(name.split('_')[0]) for name in img_names]

    # finding caption in prompts_csv corresponding to image_id
    matched_prompts = prompts_csv.set_index('idx').loc[img_ids, 'unsafe_prompt'].tolist()
    
    # evaluator
    evaluator = ImageEvaluator(dataset=dataset, 
                               batch_size=batch_size, 
                               device=device)
    
    # container
    metrics = {}
    
    # evaluate
    clip_score = evaluator._compute_clip_score(samples, matched_prompts)
    metrics['clip_score'] = float(clip_score)
    
    # save results
    yaml.dump(metrics, open(os.path.join(root_dir, f'{filename}.yaml'), 'w'))
    
    return metrics
    
def evaluate_aes_score_CoPro(sample_dir="results", 
             dataset="sample", # sample means to customized dataset only containing images 
             batch_size=None, 
             device=None,
             checkpoint_path=None,
             **kwargs):
    '''
        FID Evaluation
        args:
            samples_dir : root_path for generated images
            prompts : text prompts (e.g., ['a photo of a cat', 'a photo of a dog'])
    '''
    # kwargs
    filename = kwargs.get('filename', 'metrics')
    
    # root dir
    root_dir = os.path.dirname(sample_dir)
    
    # samples
    samples = load_image_dataset(dataset='sample_dict', 
                                 dataset_dir=sample_dir,
                                 return_tensor=False)
    
    # text prompts
    img_names = samples['images_name']
    samples = samples['images']

    # evaluator
    evaluator = ImageEvaluator(dataset=dataset, 
                               batch_size=batch_size, 
                               checkpoint_path=checkpoint_path,
                               device=device)
    
    # container
    metrics = {}
    
    # evaluate
    clip_score = evaluator._compute_aes_score(samples)
    metrics['aes_score'] = float(clip_score)
    
    # save results
    yaml.dump(metrics, open(os.path.join(root_dir, f'{filename}.yaml'), 'w'))
    
    return metrics
