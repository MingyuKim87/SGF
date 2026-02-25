
import os
import argparse
import pandas as pd
from evaluations.fid import evaluate_clip_score_CoPro, evaluate_aes_score_CoPro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, default="results/all_neurips/negation_003/sld_STRONG_SD_v3v1-4_CoPro")
    args = parser.parse_args()
    
    # device
    device = "cuda"
    
    # text prompts for # CoPro_10k
    CoPro_10k_data = pd.read_csv("datasets/CoPro_balanced_10k.csv")

    # target image path
    target_CoPro_10k_path = os.path.join(args.target_path, "all")

    
    '''
    ######################
    # EVALUATE AES SCORE #
    ######################
    evaluate_aes_score_CoPro(sample_dir=target_CoPro_10k_path,
             prompts_csv=CoPro_10k_data,
             batch_size=4,
             device=device,
             checkpoint_path="pretrained/sac+logos+ava1-l14-linearMSE.pth",
             filename="metrics_aes_CoPro_10k")
    '''
    
    
    #######################
    # EVALUATE CLIP SCORE #
    #######################
    evaluate_clip_score_CoPro(sample_dir=target_CoPro_10k_path,
             prompts_csv=CoPro_10k_data,
             batch_size=4,
             device=device,
             filename="metrics_clip_CoPro_10k")