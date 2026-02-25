# -*- coding: utf-8 -*-
"""
Select the first N classes (WNIDs) from ImageNet-1k classnames.txt, then collect:
- GEN_ROOT/<WNID>/*.jpg
- IMAGENET_VAL_ROOT/<WNID>/*
and compute a single *marginal* FID over the entire set (ignoring class labels).

- FID backbone: evaluations.utils.fid.calculate_fid (same as existing code)
- Save path (default): GEN_ROOT/metrics/metrics_fid_imagenet_top{N}.yaml
"""

import os
import argparse
import yaml
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# Import according to your project structure (use as-is)
from evaluations.base_image import ImageEvaluator  # _compute_fid에서 calculate_fid 호출
# If the path differs, e.g.: from evaluations.fid import ImageEvaluator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jpeg", ".JPEG", ".JPG", ".PNG", ".BMP", ".WEBP"}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_top_wnids(classnames_path: str, top_n: int) -> List[str]:
    """Extract only the WNIDs from the first top_n lines of classnames.txt."""
    wnids = []
    with open(classnames_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            wnid = s.split()[0]
            wnids.append(wnid)
            if len(wnids) >= top_n:
                break
    if len(wnids) < top_n:
        print(f"[WARN] Requested {top_n} classes from classnames.txt, but read only {len(wnids)}.")
    return wnids

def list_all_image_paths_under(root: str, wnids: List[str]) -> List[str]:
    """Collect all image paths under root/<WNID> for the given WNIDs."""
    paths = []
    for wnid in wnids:
        d = os.path.join(root, wnid)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(d, fn))
    paths.sort()
    return paths

def load_pil_images(filepaths: List[str]) -> List[Image.Image]:
    """Load images as a list of PIL.Image (copy then close to reduce file-handle usage)."""
    out = []
    for p in tqdm(filepaths, desc="Load images", unit="img"):
        try:
            with Image.open(p) as im:
                out.append(im.convert("RGB").copy())
        except Exception as e:
            print(f"[WARN] Failed to open {p}: {e}")
    return out

def evaluate_fid_marginal(gen_root: str,
                          real_root: str,
                          classnames_path: str,
                          num_classes: int,
                          batch_size: int,
                          device: str,
                          out_dir: Optional[str] = None,
                          filename: Optional[str] = None) -> Dict[str, float]:
    """
    Compute marginal (class-agnostic) FID over the selected WNIDs.
    """
    wnids = read_top_wnids(classnames_path, num_classes)
    print(f"[INFO] Number of target classes (WNIDs): {len(wnids)} (requested: {num_classes})")

    gen_img_root = os.path.join(gen_root, "ref")

    gen_paths  = list_all_image_paths_under(gen_img_root, wnids)
    real_paths = list_all_image_paths_under(real_root, wnids)

    if len(gen_paths) == 0:
        raise RuntimeError(f"[ERROR] No generated images found: {gen_root}/<WNID>/*.jpg")
    if len(real_paths) == 0:
        raise RuntimeError(f"[ERROR] No reference (real) images found: {real_root}/<WNID>/*")

    print(f"[INFO] Total generated images: {len(gen_paths)}, total reference images: {len(real_paths)}")

    # Load PIL images
    gen_imgs  = load_pil_images(gen_paths)
    real_imgs = load_pil_images(real_paths)

    # Use the existing backbone as-is: ImageEvaluator._compute_fid
    evaluator = ImageEvaluator(dataset="sample", dataset_root=real_root,
                               batch_size=batch_size, device=device)

    metrics = {}
    fid = evaluator._compute_fid(gen_imgs, real_imgs)
    metrics["fid"] = float(fid)

    # (Optional) If you also want KID, uncomment below
    # kid = evaluator._compute_kid(gen_imgs, real_imgs)
    # metrics["kid"] = float(kid)
    # metrics["log_kid"] = float(np.log(kid))

    # Save
    if out_dir is None:
        out_dir = os.path.join(gen_root, "metrics")
    ensure_dir(out_dir)

    if filename is None:
        filename = f"metrics_fid_imagenet_top{num_classes}"

    save_path = os.path.join(out_dir, f"{filename}.yaml")
    with open(save_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"[SAVE] {save_path}")

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", type=str, required=True,
                    help="Root directory of generated images (GEN_ROOT/<WNID>/*.jpg)")
    ap.add_argument("--imagenet_val_root", type=str, required=True,
                    help="Root directory of ImageNet val images (VAL_ROOT/<WNID>/*)")
    ap.add_argument("--classnames", type=str, required=True,
                    help="classnames.txt (first token is WNID)")
    ap.add_argument("--num_classes", type=int, default=500,
                    help="Use only the first N classes")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--filename", type=str, default=None)
    args = ap.parse_args()

    _ = evaluate_fid_marginal(
        gen_root=args.gen_root,
        real_root=args.imagenet_val_root,
        classnames_path=args.classnames,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        filename=args.filename,
    )

if __name__ == "__main__":
    main()
