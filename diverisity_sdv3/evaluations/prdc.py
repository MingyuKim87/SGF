# evaluations/prdc.py
# -*- coding: utf-8 -*-
"""
Precision/Recall/Density/Coverage evaluator for image generation
- Precision/Recall: Kynkäännienmi et al., 2019 (Improved Precision and Recall…)
- Density/Coverage: Naeem et al., 2020 (Reliable Fidelity and Diversity Metrics…)
Backbone: Inception-V3 (pool3, 2048-D) by default.

Usage:
from evaluations.prdc import evaluate_prdc
evaluate_prdc(sample_dir=..., dataset_root=..., batch_size=64, device="cuda", k=5, filename="metrics_prdc")

Outputs:
- {filename}.json saved under sample_dir (precision, recall, density, coverage, counts, k, backbone, feature_dim)
"""

import os
import json
import math
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from typing import Optional
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
except Exception:
    from torchvision.transforms.functional import InterpolationMode
from torchvision.models import inception_v3, Inception_V3_Weights


# -----------------------
# I. Image loading
# -----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dp, fn))
    paths.sort()
    return paths

# --- PATCH: add helper to build preprocess ---
def _resize_op(size):
    # torchvision 버전별 'antialias' 유무 대응
    try:
        return transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    except TypeError:
        return transforms.Resize(size, interpolation=InterpolationMode.BICUBIC)

def build_inception_preprocess(mode: str = "squash", size: int = 299):
    """
    mode:
      - "squash": 정확히 (299,299)로 변환 (no crop). 512→299 같은 다운스케일에 안전.
      - "crop":   짧은 변을 299로 맞춘 뒤 CenterCrop(299).
      - "torchvision": torchvision weights.transforms() 그대로 사용.
    """
    weights = Inception_V3_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
    std  = weights.meta.get("std",  (0.229, 0.224, 0.225))

    if mode.lower() == "squash":
        return transforms.Compose([
            _resize_op((size, size)),        # (H,W) 정확 축소
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif mode.lower() == "crop":
        return transforms.Compose([
            _resize_op(size),                # 짧은 변을 size로
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif mode.lower() == "torchvision":
        return weights.transforms()
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

class ImageListDataset(Dataset):
    def __init__(self, files: List[str], transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        with Image.open(fp) as img:
            img = img.convert("RGB")
        return self.transform(img)


# -----------------------
# II. Feature extractor
# -----------------------
class InceptionPool3(nn.Module):
    """Inception-V3 feature extractor that returns 2048-D pool3 features."""
    def __init__(self, preprocess: Optional[nn.Module] = None):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1

        # pretrained weights는 aux_logits=True 아키텍처를 기대합니다.
        self.model = inception_v3(weights=weights, aux_logits=True)
        self.model.dropout = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 전처리: 명시적으로 299×299로 맞추도록 기본값 'squash'
        self.preprocess = preprocess if preprocess is not None else build_inception_preprocess("squash", 299)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model(x) now returns the 2048-D vector after avgpool (since fc is Identity)
        out = self.model(x)

        # fc=Identity 이므로 'logits'가 곧 2048-D 특징입니다.
        if hasattr(out, "logits"):        # InceptionOutputs(namedtuple) 대비
            out = out.logits
        elif isinstance(out, (tuple, list)):  # tuple로 나올 가능성 대비
            out = out[0]
        return out  # [B, 2048]

# --- PATCH: plumb preprocess choice through feature extractor ---
def build_dataloader(image_dir: str, batch_size: int, num_workers: int, preprocess) -> DataLoader:
    files = list_images(image_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found under: {image_dir}")
    ds = ImageListDataset(files, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl, len(files)

@torch.no_grad()
def extract_inception_features(
    image_dir: str,
    batch_size: int,
    device: str,
    num_workers: int = 4,
    preprocess_mode: str = "squash",     # <-- NEW: 기본 'squash'로 정확히 299×299
    input_size: int = 299,               # <-- NEW
) -> np.ndarray:
    preprocess = build_inception_preprocess(preprocess_mode, input_size)
    model = InceptionPool3(preprocess=preprocess).to(device)
    dl, n = build_dataloader(image_dir, batch_size, num_workers, model.preprocess)

    feats = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        f = model(x)  # [B, 2048]
        feats.append(f.float().cpu())
    feats = torch.cat(feats, dim=0).numpy()
    assert feats.shape[0] == n
    return feats  # [N, 2048]


# -----------------------
# III. Distance helpers (blockwise)
# -----------------------
def _blockwise_cdist_min(A: torch.Tensor, B: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """
    For each row in A, returns the minimum L2 distance to any row in B.
    Shapes: A=[NA,D], B=[NB,D]
    Returns: mins [NA]
    """
    device = A.device
    NA = A.shape[0]
    mins = torch.full((NA,), float("inf"), device=device)
    for i in range(0, NA, block):
        Ai = A[i:i+block]  # [b, D]
        d = torch.cdist(Ai, B)  # [b, NB]
        mi = d.min(dim=1).values
        mins[i:i+block] = mi
    return mins


def _self_knn_radii(X: torch.Tensor, k: int, block: int = 512) -> torch.Tensor:
    """
    k-th nearest neighbor distance within X for each row (excluding self).
    X: [N, D] on device
    Returns: radii [N]
    """
    N = X.shape[0]
    if N <= 1:
        return torch.zeros((N,), device=X.device)

    k = min(k, max(1, N - 1))
    radii = torch.empty((N,), device=X.device)
    INF = 1e10

    for i in range(0, N, block):
        Xi = X[i:i+block]                              # [b, D]
        d = torch.cdist(Xi, X)                         # [b, N]
        # mask diagonal for current block rows
        rows = torch.arange(i, min(i+block, N), device=X.device)
        cols = torch.arange(0, Xi.shape[0], device=X.device)
        d[cols, rows] = INF
        # take (k)-th smallest (already excludes self)
        vals, _ = torch.topk(d, k=k, dim=1, largest=False)
        # k-th neighbor among others is index k-1
        radii[i:i+Xi.shape[0]] = vals[:, k-1]
    return radii


def _precision_and_density(gen: torch.Tensor, real: torch.Tensor, real_radii: torch.Tensor, k: int, block: int = 1024) -> Tuple[float, float]:
    """
    Precision: fraction of generated points that fall into union of real k-NN balls.
    Density: average number of real balls that contain a generated point, normalized by k.
    """
    NG = gen.shape[0]
    inside = torch.zeros((NG,), dtype=torch.bool, device=gen.device)
    density_acc = torch.zeros((NG,), dtype=torch.float32, device=gen.device)

    # Compare G (rows) to R (cols)
    for i in range(0, NG, block):
        Gi = gen[i:i+block]                     # [b, D]
        d = torch.cdist(Gi, real)               # [b, NR]
        # real_radii: [NR] -> broadcast to [b, NR]
        within = d <= real_radii.unsqueeze(0)   # [b, NR]
        counts = within.sum(dim=1)              # [b]
        density_acc[i:i+Gi.shape[0]] = counts.float() / max(1, k)
        inside[i:i+Gi.shape[0]] = counts > 0

    precision = inside.float().mean().item()
    density = density_acc.mean().item()
    return precision, density


def _coverage(real: torch.Tensor, gen: torch.Tensor, real_radii: torch.Tensor, block: int = 1024) -> float:
    """
    Coverage: fraction of real points for which nearest generated neighbor is within its real k-NN radius.
    """
    NR = real.shape[0]
    covered = torch.zeros((NR,), dtype=torch.bool, device=real.device)
    for i in range(0, NR, block):
        Ri = real[i:i+block]                  # [b, D]
        d = torch.cdist(Ri, gen)              # [b, NG]
        nearest = d.min(dim=1).values         # [b]
        covered[i:i+Ri.shape[0]] = nearest <= real_radii[i:i+Ri.shape[0]]
    return covered.float().mean().item()


def _recall(real: torch.Tensor, gen: torch.Tensor, gen_radii: torch.Tensor, block: int = 1024) -> float:
    """
    Recall: fraction of real points that are inside the union of generated k-NN balls.
    """
    NR = real.shape[0]
    recalled = torch.zeros((NR,), dtype=torch.bool, device=real.device)
    for i in range(0, NR, block):
        Ri = real[i:i+block]                    # [b, D]
        d = torch.cdist(Ri, gen)                # [b, NG]
        within = d <= gen_radii.unsqueeze(0)    # [b, NG]
        recalled[i:i+Ri.shape[0]] = within.any(dim=1)
    return recalled.float().mean().item()

def compute_prdc_from_features(
    real_feats_np: np.ndarray,
    gen_feats_np:  np.ndarray,
    k: int = 5,
    device: str = "cuda",
):
    """
    PRDC를 '이미 추출해 둔 특징'에서 바로 계산.
    Args:
        real_feats_np: [NR, D] numpy float32
        gen_feats_np:  [NG, D] numpy float32
        k: k-NN 반경 계산용 k (각 세트 크기에 맞춰 자동 클램핑)
        device: "cuda" or "cpu"
    Returns:
        dict(precision, recall, density, coverage, k_used_real, k_used_gen, n_real, n_gen)
    """
    assert real_feats_np.ndim == 2 and gen_feats_np.ndim == 2
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    real = torch.from_numpy(real_feats_np).to(device_t, non_blocking=True)
    gen  = torch.from_numpy(gen_feats_np).to(device_t, non_blocking=True)

    NR, NG = real.shape[0], gen.shape[0]
    k_real = min(max(1, NR - 1), max(1, k))
    k_gen  = min(max(1, NG - 1), max(1, k))

    real_radii = _self_knn_radii(real, k=k_real, block=512)
    gen_radii  = _self_knn_radii(gen,  k=k_gen,  block=512)

    precision, density = _precision_and_density(gen, real, real_radii, k=k_real, block=1024)
    coverage = _coverage(real, gen, real_radii, block=1024)
    recall   = _recall(real,  gen, gen_radii,  block=1024)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "k_used_real": int(k_real),
        "k_used_gen": int(k_gen),
        "n_real": int(NR),
        "n_gen": int(NG),
    }


# -----------------------
# IV. Public API
# -----------------------
def evaluate_prdc(
    sample_dir: str,
    dataset_root: str,
    batch_size: int = 64,
    device: str = "cuda",
    k: int = 5,
    filename: str = "metrics_prdc",
    num_workers: int = 4
):
    """
    Compute Precision, Recall, Density, Coverage between generated samples and a reference dataset.

    Args:
        sample_dir: path to generated images
        dataset_root: path to reference/real images
        batch_size: batch size for feature extraction
        device: "cuda" or "cpu"
        k: k in k-NN (default 5). If a set is too small, k is clamped to (n-1).
        filename: output json filename (saved under sample_dir)
        num_workers: dataloader workers
    """
    assert os.path.isdir(sample_dir), f"Not a directory: {sample_dir}"
    assert os.path.isdir(dataset_root), f"Not a directory: {dataset_root}"

    # 1) Feature extraction
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    real_feats = extract_inception_features(dataset_root, batch_size=batch_size, device=str(device_t))   # [NR, 2048]
    gen_feats  = extract_inception_features(sample_dir,     batch_size=batch_size, device=str(device_t))   # [NG, 2048]

    NR, NG = real_feats.shape[0], gen_feats.shape[0]
    k_real = min(k, max(1, NR - 1))
    k_gen  = min(k, max(1, NG - 1))

    # 2) Torch tensors on device
    real = torch.from_numpy(real_feats).to(device_t, non_blocking=True)
    gen  = torch.from_numpy(gen_feats).to(device_t, non_blocking=True)

    # 3) k-NN radii within each set
    real_radii = _self_knn_radii(real, k=k_real, block=512)  # [NR]
    gen_radii  = _self_knn_radii(gen,  k=k_gen,  block=512)  # [NG]

    # 4) Metrics
    precision, density = _precision_and_density(gen, real, real_radii, k=k_real, block=1024)
    coverage = _coverage(real, gen, real_radii, block=1024)
    recall   = _recall(real,  gen, gen_radii,  block=1024)

    # 5) Save
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "k": int(k),
        "n_real": int(NR),
        "n_gen": int(NG),
        "backbone": "inception_v3_pool3",
        "feature_dim": int(real_feats.shape[1]),
    }
    out_path = os.path.join(sample_dir, f"{filename}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[PRDC] Saved metrics to: {out_path}\n{out}")
    return out


if __name__ == "__main__":
    m = InceptionPool3().eval()
    x = torch.randn(2, 3, 299, 299)  # transforms()도 299 크기를 기대
    with torch.no_grad():
        f = m(x)
    print(f.shape)

    