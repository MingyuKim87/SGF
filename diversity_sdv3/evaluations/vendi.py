# evaluations/vendi.py
# -*- coding: utf-8 -*-
"""
Vendi Score (Friedman & Dieng, 2023) for a single set of images (typically generated set).
Default kernel: cosine; optional RBF (bandwidth via median heuristic).
Backbone: Inception-V3 pool3 by default (same as PRDC). You may switch to CLIP/DINO if you prefer.

Usage:
from evaluations.vendi import evaluate_vendi
evaluate_vendi(sample_dir=..., batch_size=64, device="cuda", kernel="cosine", filename="metrics_vendi")
"""

import os
import json
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from evaluations.prdc import extract_inception_features  # reuse


def _pairwise_sq_dists_blockwise(X: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """
    Returns full NxN squared distance matrix (on CPU) in blocks to control memory.
    NOTE: For very large N, consider subsampling via max_samples.
    """
    N, D = X.shape
    device = X.device
    K = torch.empty((N, N), dtype=torch.float32, device=device)
    for i in range(0, N, block):
        Xi = X[i:i+block]               # [b, D]
        d = torch.cdist(Xi, X, p=2)     # [b, N]
        K[i:i+Xi.shape[0], :] = d * d
    return K


def _cosine_gram(X: torch.Tensor) -> torch.Tensor:
    # Normalize rows to unit length and compute Gram = X X^T
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    return X @ X.t()


def vendi_score_from_gram(K: torch.Tensor) -> float:
    """
    Vendi = exp(H(p)), p = normalized eigenvalues of PSD Gram matrix K.
    """
    # symmetrize & ensure numeric stability
    K = 0.5 * (K + K.t())
    # eigenvalues (symmetric)
    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=0)
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    p = eigvals / s
    H = -(p * (p + 1e-12).log()).sum()
    vendi = torch.exp(H).item()
    return float(vendi)

def vendi_from_features(
    feats_np: np.ndarray,
    device: str = "cuda",
    kernel: str = "cosine",
    max_samples: int = 5000
) -> float:
    """
    이미 추출한 특징으로 Vendi 계산
    """
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    X = torch.from_numpy(feats_np).to(device_t)
    N = X.shape[0]
    if (max_samples is not None) and (N > max_samples):
        idx = np.random.RandomState(0).choice(N, size=max_samples, replace=False)
        X = X[idx]

    if kernel.lower() == "cosine":
        K = _cosine_gram(X)
    elif kernel.lower() == "rbf":
        D2 = _pairwise_sq_dists_blockwise(X, block=max(1, 4096 // max(1, X.shape[0] // 1024)))
        triu = torch.triu(D2, diagonal=1)
        sigma2 = triu[triu > 0].median()
        sigma2 = sigma2 if torch.isfinite(sigma2) and (sigma2 > 0) else D2.mean().clamp(min=1e-6)
        K = torch.exp(-D2 / (2.0 * sigma2))
        del D2
    else:
        raise ValueError("kernel must be 'cosine' or 'rbf'.")

    return vendi_score_from_gram(K)


def evaluate_vendi(
    sample_dir: str,
    batch_size: int = 64,
    device: str = "cuda",
    kernel: str = "cosine",            # "cosine" or "rbf"
    filename: str = "metrics_vendi",
    num_workers: int = 4,
    max_samples: Optional[int] = 5000  # to limit O(N^2) memory/time for huge sets
):
    """
    Compute Vendi score on the generated set in sample_dir.

    Args:
        sample_dir: directory with generated images
        batch_size: feature extraction batch size
        device: "cuda" or "cpu"
        kernel: "cosine" (default, cheaper) or "rbf" (median heuristic bandwidth)
        filename: output json name (saved under sample_dir)
        max_samples: if not None, randomly subsample at most this many images before computing Gram
    """
    assert os.path.isdir(sample_dir), f"Not a directory: {sample_dir}"

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    feats_np = extract_inception_features(sample_dir, batch_size=batch_size, device=str(device_t))
    N = feats_np.shape[0]

    if (max_samples is not None) and (N > max_samples):
        idx = np.random.RandomState(0).choice(N, size=max_samples, replace=False)
        feats_np = feats_np[idx]
        N = feats_np.shape[0]

    X = torch.from_numpy(feats_np).to(device_t)

    if kernel.lower() == "cosine":
        K = _cosine_gram(X)  # [N, N]
    elif kernel.lower() == "rbf":
        # median heuristic for bandwidth
        D2 = _pairwise_sq_dists_blockwise(X, block=max(1, 4096 // max(1, X.shape[0] // 1024)))
        # avoid diagonal zeros in median; use upper triangle excluding diag
        triu = torch.triu(D2, diagonal=1)
        sigma2 = triu[triu > 0].median()
        sigma2 = sigma2 if torch.isfinite(sigma2) and (sigma2 > 0) else D2.mean().clamp(min=1e-6)
        K = torch.exp(-D2 / (2.0 * sigma2))
        del D2
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'cosine' or 'rbf'.")

    vendi = vendi_score_from_gram(K)

    out = {
        "vendi": float(vendi),
        "n_samples": int(N),
        "kernel": kernel,
        "backbone": "inception_v3_pool3",
        "feature_dim": int(feats_np.shape[1]),
        "max_samples": int(max_samples) if max_samples is not None else None,
    }
    out_path = os.path.join(sample_dir, f"{filename}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Vendi] Saved metrics to: {out_path}\n{out}")
    return out
