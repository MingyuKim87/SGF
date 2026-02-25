
import os
import argparse
import pandas as pd
from evaluations.fid import evaluate_fid, evaluate_clip_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, default="results/coco30k/negation_003/sld_rep_MEDIUM_SDv1-4_coco30k_v2")
    args = parser.parse_args()
    
    # device
    device = "cuda"
    
    # CoProV2
    coco30k_10k_img_path = "datasets/coco30k_10k_img"

    # text prompts for coco30k_10k
    coco30k_10k_data = pd.read_csv("datasets/coco_30k_10k.csv")

    # sd14_coco30k_10k
    sd14_coco30k_10k_path = os.path.join("results/coco30k/vanilla/std_SDv1-4_ring-a-bell", "all")
    
    # target image path
    target_coco30k_10k_path = os.path.join(args.target_path, "all")
    
    
    ################
    # EVALUATE FID #
    ################
    # evaluate_fid(sample_dir=target_coco30k_10k_path,
    #          dataset_root=sd14_coco30k_10k_path,
    #          batch_size=4,
    #          device=device,
    #          filename="metrics_sd14_coco30k_10k")

    
    # evaluate fid
    evaluate_fid(sample_dir=target_coco30k_10k_path,
             dataset_root=coco30k_10k_img_path,
             batch_size=4,
             device=device,
             filename="metrics_org_coco30k_10k")

    #######################
    # EVALUATE CLIP SCORE #
    #######################
    evaluate_clip_score(sample_dir=target_coco30k_10k_path,
             prompts_csv=coco30k_10k_data,
             batch_size=4,
             device=device,
             filename="metrics_clip_coco30k_10k")