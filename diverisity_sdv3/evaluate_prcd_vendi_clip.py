import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json
import csv
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

# Reuse: Inception feature extraction is implemented in the prdc module extractor
from evaluations.prdc import extract_inception_features, compute_prdc_from_features
from evaluations.vendi import vendi_from_features

# ---- CLIP score (torchmetrics) ----
from torchmetrics.multimodal.clip_score import CLIPScore

# ---- AES (use the provided class as-is; the import path may differ, so keep a try-chain if needed) ----
from evaluations.utils.aes import AE

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# tools/eval_imagenet_subset.py
"""
Select only the first N classes (default 500) from classnames.txt in ImageNet-1k and compute:
- Per-class PRDC (Precision, Recall, Density, Coverage)
- Per-class Vendi
- Marginal FID (FID between overall distributions, ignoring class labels)

This follows the paper setup (Section 5.1) exactly:
- "Measure per class/prompt and then average" (PRDC, Vendi)
- Measure "marginal FID" separately
- The reference set is the ImageNet-1k **validation set**.
"""

import os
import argparse
import json
import csv
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

# Reuse: Inception feature extraction is implemented in the prdc module extractor
from evaluations.prdc import extract_inception_features, compute_prdc_from_features
from evaluations.vendi import vendi_from_features

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_wnids_from_classnames(path: str, top_n: int) -> List[str]:
    wnids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]  # 첫 토큰을 WNID로 사용 (예: n01440764)
            wnids.append(token)
            if len(wnids) >= top_n:
                break
    if len(wnids) < top_n:
        print(f"[WARN] Requested {top_n} classes from classnames.txt, but found only {len(wnids)}.")
    return wnids

def read_wnid_to_name(path: str, top_n: int) -> Dict[str, str]:
    """
    classnames.txt: "n01440764 tench, Tinca tinca ..." → {'n01440764': 'tench'}
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(maxsplit=1)
            wnid = parts[0]
            cname = wnid
            if len(parts) > 1:
                cname = parts[1].split(",")[0].strip().replace("_", " ")
            mapping[wnid] = cname
            if len(mapping) >= top_n:
                break
    return mapping

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def class_dir(root: str, wnid: str) -> Optional[str]:
    """Return root/wnid if it exists as a directory; otherwise return None."""
    cand = os.path.join(root, wnid)
    return cand if os.path.isdir(cand) else None

def list_images(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dp, fn))
    files.sort()
    return files

def load_pil_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        except Exception as e:
            print(f"[WARN] Failed to open {p}: {e}")
    return imgs

def extract_or_cache_features(dir_path: str,
                              batch_size: int,
                              device: str,
                              cache_root: Optional[str],
                              cache_key: Optional[str],
                              preprocess_mode: str = "squash",
                              input_size: int = 299) -> np.ndarray:
    if cache_root and cache_key:
        ensure_dir(cache_root)
        npy_path = os.path.join(cache_root, f"{cache_key}.npy")
        if os.path.isfile(npy_path):
            return np.load(npy_path)

    feats = extract_inception_features(
        dir_path, batch_size=batch_size, device=device,
        preprocess_mode=preprocess_mode, input_size=input_size
    )
    if cache_root and cache_key:
        np.save(os.path.join(cache_root, f"{cache_key}.npy"), feats)
    return feats

# ---------------------------
# CLIP / AES calculators
# ---------------------------
@torch.no_grad()
def compute_clip_score_for_images(
    images: List[Image.Image],
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 64,
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """
    Same approach as the provided code:
    - Use torchmetrics.CLIPScore
    - Apply transforms.ToTensor() and then scale by x*255
    """
    if len(images) == 0:
        return float("nan")
    metric = CLIPScore(model_name_or_path=model_name).to(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])

    for i in range(0, len(images), batch_size):
        batch_imgs = [tf(img).to(device) for img in images[i : i + batch_size]]
        batch_txts = texts[i : i + batch_size]
        metric.update(images=batch_imgs, text=batch_txts)

    return float(metric.compute().item())


@torch.no_grad()
def compute_aes_score_for_images(
    images: List[Image.Image],
    device: str = "cuda",
    batch_size: int = 64,
    checkpoint_path: Optional[str] = None,
) -> float:
    """
    Use the provided AE class as-is (checkpoint required).
    """
    if (checkpoint_path is None) or (AE is None):
        return float("nan")
    if len(images) == 0:
        return float("nan")

    ae = AE(path=checkpoint_path, device=device)
    scores = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        s = ae(chunk)  # AE가 np.ndarray 또는 list 반환
        scores.append(s)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return float(scores.mean()), scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", type=str, required=True,
                    help="Root directory of generated images (per-class WNID subfolders)")
    ap.add_argument("--imagenet_val_root", type=str, required=True,
                    help="Root directory of ImageNet validation set (per-class WNID subfolders)")
    ap.add_argument("--classnames", type=str, required=True,
                    help="classnames.txt where the first token of each line is a WNID")
    ap.add_argument("--num_classes", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k", type=int, default=5, help="PRDC의 k")
    ap.add_argument("--vendi_kernel", type=str, default="cosine", choices=["cosine", "rbf"])
    ap.add_argument("--max_vendi_samples", type=int, default=5000)
    ap.add_argument("--real_cache_dir", type=str, default=None,
                    help="Directory to cache real (ImageNet val) features (optional)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Directory to save metrics (default: target_path/metrics)")
    ap.add_argument("--resize_mode", type=str, default="squash",
                choices=["squash", "crop", "torchvision"],
                help="How to resize inputs to 299x299 for Inception (default: squash = exact resize)")
    ap.add_argument("--input_size", type=int, default=299)
    ap.add_argument("--compute_clip", action="store_true",
                    help="Compute per-class CLIP score using prompt 'a photo of {classname}")
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--compute_aes", action="store_true",
                    help="Compute per-class Aesthetic score")
    ap.add_argument("--aes_checkpoint", type=str, default=None,
                    help="Path to AES checkpoint (.pth) (e.g., sac+logos+ava1-l14-linearMSE.pth)")
    args = ap.parse_args()

    device = args.device
    wnids = read_wnids_from_classnames(args.classnames, args.num_classes)
    wnid2name = read_wnid_to_name(args.classnames, args.num_classes)

    out_dir = args.out_dir or os.path.join(args.target_path, "metrics")
    ensure_dir(out_dir)

    # Storage for per-class results
    rows = []

    # Global accumulators for overall CLIP/AES average
    overall_clip_metric = None
    overall_clip_count = 0
    if args.compute_clip:
        overall_clip_metric = CLIPScore(model_name_or_path=args.clip_model).to(device)
         # Same preprocessing as provided code
        clip_tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])

    overall_aes_sum: float = 0.0
    overall_aes_count = 0

    missing = []

    print(f"[INFO] Requested number of classes: {args.num_classes}, actually loaded: {len(wnids)}")

    for wnid in tqdm(wnids, desc="Per-class metrics"):
        real_dir = class_dir(args.imagenet_val_root, wnid)
        gen_dir  = class_dir(os.path.join(args.target_path, "ref"), wnid)

        if (real_dir is None) or (gen_dir is None):
            missing.append(wnid)
            continue

         # Feature extraction (caching is recommended for real)
        real_feats = extract_or_cache_features(
            real_dir,
            batch_size=args.batch_size,
            device=device,
            cache_root=args.real_cache_dir,
            cache_key=f"real_{wnid}",
            preprocess_mode=args.resize_mode, 
            input_size=args.input_size
        )
        gen_feats = extract_inception_features(
            gen_dir,
            batch_size=args.batch_size,
            device=device,
            preprocess_mode=args.resize_mode, 
            input_size=args.input_size
        )

         # PRDC (per class)
        prdc = compute_prdc_from_features(
            real_feats, gen_feats, k=args.k, device=device
        )

        # Vendi (per class, generation set only)
        vendi = vendi_from_features(
            gen_feats, device=device, kernel=args.vendi_kernel, max_samples=args.max_vendi_samples
        )

        row = {
            "wnid": wnid,
            "classname": wnid2name.get(wnid, wnid),
            "n_real": prdc["n_real"],
            "n_gen": prdc["n_gen"],
            "precision": prdc["precision"],
            "recall": prdc["recall"],
            "density": prdc["density"],
            "coverage": prdc["coverage"],
            "vendi": vendi,
        }

        # ----- (Optional) CLIP score: "a photo of {classname}" -----
        if args.compute_clip:
            gen_paths = list_images(gen_dir)
            gen_imgs = load_pil_images(gen_paths)
            prompts = [f"a photo of {wnid2name.get(wnid, wnid)}"] * len(gen_imgs)

            # per-class
            clip_cls = compute_clip_score_for_images(
                gen_imgs, prompts, device=device, batch_size=args.batch_size, model_name=args.clip_model
            )
            row["clip_score"] = clip_cls

            # overall (accumulate into global metric)
            for i in range(0, len(gen_imgs), args.batch_size):
                batch_imgs = [clip_tf(img).to(device) for img in gen_imgs[i : i + args.batch_size]]
                batch_txts = prompts[i : i + args.batch_size]
                overall_clip_metric.update(images=batch_imgs, text=batch_txts)
            overall_clip_count += len(gen_imgs)

         # ----- (Optional) AES score -----
        if args.compute_aes and (args.aes_checkpoint is not None):
            gen_paths = list_images(gen_dir)
            gen_imgs = load_pil_images(gen_paths)
            aes_cls, aes = compute_aes_score_for_images(
                gen_imgs, device=device, batch_size=args.batch_size, checkpoint_path=args.aes_checkpoint
            )
            row["aes_score"] = aes_cls

            # overall
            aes = np.asarray(aes, dtype=np.float32).reshape(-1)
            overall_aes_sum += float(aes.sum())
            overall_aes_count += max(1, len(gen_imgs))

        rows.append(row)


    # Missing report
    if missing:
        print(f"[WARN] Skipped WNIDs due to missing real or gen folder (total {len(missing)}):")
        print(", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    # Save per-class CSV
    per_class_csv = os.path.join(out_dir, f"imagenet_top{args.num_classes}_per_class_metrics.csv")
    df = pd.DataFrame(rows)
    df.to_csv(per_class_csv, index=False)
    print(f"[SAVE] per-class metrics -> {per_class_csv}")

    # Macro average (average across classes)
    summary = {}
    if len(df) > 0:
        summary = {
            "num_classes_requested": int(args.num_classes),
            "num_classes_evaluated": int(len(df)),
            "macro_precision": float(df["precision"].mean()),
            "macro_recall": float(df["recall"].mean()),
            "macro_density": float(df["density"].mean()),
            "macro_coverage": float(df["coverage"].mean()),
            "macro_vendi": float(df["vendi"].mean()),
            "macro_precision_std": float(df["precision"].std(ddof=0)),
            "macro_recall_std": float(df["recall"].std(ddof=0)),
            "macro_density_std": float(df["density"].std(ddof=0)),
            "macro_coverage_std": float(df["coverage"].std(ddof=0)),
            "macro_vendi_std": float(df["vendi"].std(ddof=0)),
        }
        if args.compute_clip and ("clip_score" in df.columns):
            summary["macro_clip"] = float(df["clip_score"].mean())
            summary["macro_clip_std"] = float(df["clip_score"].std(ddof=0))
        if args.compute_aes and ("aes_score" in df.columns):
            summary["macro_aes"] = float(df["aes_score"].mean())
            summary["macro_aes_std"] = float(df["aes_score"].std(ddof=0))

     # Overall CLIP/AES (averaged over all images)
    if args.compute_clip and (overall_clip_metric is not None) and (overall_clip_count > 0):
        summary["clip_overall"] = float(overall_clip_metric.compute().item())
    if args.compute_aes and (overall_aes_count > 0):
        summary["aes_overall"] = float(overall_aes_sum / overall_aes_count)

     # Save summary JSON
    summary_json = os.path.join(out_dir, f"imagenet_top{args.num_classes}_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] summary -> {summary_json}")

if __name__ == "__main__":
    main()