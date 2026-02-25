import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import torch
import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 스타일 =====
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["legend.fontsize"] = 12

LINEWIDTH  = 2.0
MARKERSIZE = 6

def load_top1_from_similarity(sim_pth: Path) -> np.ndarray:
    """
    similarity.pth  (shape: [N_q, N_val]) 로부터
    각 query의 Top-1 value 유사도만 뽑아 1D numpy로 반환.
    """
    T = torch.load(sim_pth, map_location="cpu")
    if not isinstance(T, torch.Tensor):
        raise TypeError(f"{sim_pth} should contain a torch.Tensor, got {type(T)}")
    if T.ndim != 2:
        raise ValueError(f"{sim_pth} must be 2D, got shape {tuple(T.shape)}")
    # 각 행(query)에 대해 Top-1 (정렬 보장)
    top1 = torch.topk(T, k=1, dim=1, largest=True, sorted=True).values.squeeze(1)
    return top1.numpy()

def load_bg_from_similarity_wtrain(bg_pth: Path) -> np.ndarray:
    """
    similarity_wtrain.pth (shape: [N_val, N_val]) 로부터
    self-match(자기 자신) 제외 Top-1(=상위2 중 2등) 유사도만 1D numpy로 반환.
    """
    T = torch.load(bg_pth, map_location="cpu")
    if not isinstance(T, torch.Tensor):
        raise TypeError(f"{bg_pth} should contain a torch.Tensor, got {type(T)}")
    if T.ndim != 2:
        raise ValueError(f"{bg_pth} must be 2D, got shape {tuple(T.shape)}")
    if T.shape[1] < 2:
        raise ValueError("similarity_wtrain must have at least 2 columns to drop self-match.")
    # 각 행(value)에 대해 상위 2개를 정렬, 두 번째 값을 선택
    top2 = torch.topk(T, k=2, dim=1, largest=True, sorted=True).values
    nn1  = top2[:, 1]
    return nn1.numpy()

def make_bins(data_list, bin_width=0.005, clip_range=None):
    """
    여러 분포를 모두 커버하는 bin 경계 생성.
    - bin_width: 간격
    - clip_range: (lo, hi) 튜플로 강제 범위 지정. 없으면 데이터에서 자동 결정.
    """
    if clip_range is None:
        lo = min(float(np.min(d)) for d in data_list)
        hi = max(float(np.max(d)) for d in data_list)
        # 안전 여유
        lo -= 1e-6
        hi += 1e-6
    else:
        lo, hi = clip_range
    # np.arange로 정확한 bin 폭 유지
    return np.arange(lo, hi + bin_width, bin_width)

def plot_hist(ax, data, bins, label, color, filled=True):
    """
    한 분포를 스타일 있게 히스토그램으로 그림 (density=True).
    """
    if filled:
        ax.hist(
            data, bins=bins, density=True,
            histtype='stepfilled', alpha=0.35,
            edgecolor=color, facecolor=color, linewidth=LINEWIDTH, label=label
        )
        # 외곽선 한 번 더 그려 또렷하게
        ax.hist(
            data, bins=bins, density=True,
            histtype='step', linewidth=LINEWIDTH, color=color
        )
    else:
        ax.hist(
            data, bins=bins, density=True,
            histtype='step', linewidth=LINEWIDTH, color=color, label=label
        )


def draw_combined(baseline, proposed, bg, bins, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    # 색상: 샘플 이미지 느낌에 맞춘 오렌지/그린 + 제안 모델은 블루
    plot_hist(ax, bg,       bins, label="train–train (bg)",        color="#2ca02c")  # green
    plot_hist(ax, baseline, bins, label="baseline (gen–train)",     color="#ff7f0e")  # orange
    plot_hist(ax, proposed, bins, label="proposed (gen–train)",     color="#1f77b4")  # blue

    ax.set_xlabel("Similarity (cosine)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def draw_pair(one, bg, bins, out_path: Path, label_one: str, color_one: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    plot_hist(ax, bg,  bins, label="train–train (bg)", color="#2ca02c")
    plot_hist(ax, one, bins, label=label_one,         color=color_one)
    ax.set_xlabel("Similarity (cosine)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Compare similarity distributions (hist, density).")
    # p.add_argument("--baseline_sim", type=Path, required=True,
    #                help="기존 모델의 similarity.pth 경로 (shape [N_q, N_val])")
    # p.add_argument("--proposed_sim", type=Path, required=True,
    #                help="제안 모델의 similarity.pth 경로 (shape [N_q, N_val])")
    # p.add_argument("--bg_sim", type=Path, required=True,
    #                help="Imagenette 등에서 만든 similarity_wtrain.pth 경로 (shape [N_val, N_val])")
    p.add_argument("--bin-width", type=float, default=0.005, help="히스토그램 bin 간격")
    p.add_argument("--clip", type=float, nargs=2, default=None,
                   metavar=("LO","HI"),
                   help="x축을 [LO, HI]로 고정 (기본: 데이터에서 자동 산출)")
    p.add_argument("--mode", choices=["combined", "pair"], default="combined",
                   help="combined=세 분포 한 그림, pair=baseline/bg & proposed/bg 두 그림")
    # p.add_argument("--out", type=Path, default=Path("./similarity_hist.pdf"),
    #                help="저장 파일(또는 prefix; pair 모드에서는 _baseline.pdf/_proposed.pdf로 저장)")
    args = p.parse_args()

    args.baseline_sim = "ret_plots/imagenette10_frozentext/20250918032056/similarity.pth"
    args.bg_sim = "ret_plots/imagenette10_frozentext/20250918034027/similarity_wtrain.pth"
    args.proposed_sim = "ret_plots/imagenette10_frozentext/20250918034027/similarity.pth"
    args.out = "ret_plots/similarity_hist.pdf"
    
    
    # ----- 데이터 로딩 -----
    baseline_top1 = load_top1_from_similarity(args.baseline_sim)  # gen–train (기존)
    proposed_top1 = load_top1_from_similarity(args.proposed_sim)  # gen–train (제안)
    bg_top1       = load_bg_from_similarity_wtrain(args.bg_sim)   # train–train (자기 자신 제외)

    # ----- bins -----
    bins = make_bins([baseline_top1, proposed_top1, bg_top1],
                     bin_width=args.bin_width,
                     clip_range=tuple(args.clip) if args.clip else None)

    # ----- 그리기 -----
    if args.mode == "combined":
        draw_combined(baseline_top1, proposed_top1, bg_top1, bins, args.out)
    else:
        # pair 모드: 샘플 이미지처럼 두 분포만 보이게 각각 저장
        out_base = args.out.with_name(args.out.stem + "_baseline.pdf")
        out_prop = args.out.with_name(args.out.stem + "_proposed.pdf")
        draw_pair(baseline_top1, bg_top1, bins, out_base,
                  label_one="baseline (gen–train)", color_one="#ff7f0e")  # orange
        draw_pair(proposed_top1, bg_top1, bins, out_prop,
                  label_one="proposed (gen–train)", color_one="#1f77b4")  # blue)

    print("Saved:", args.out if args.mode=="combined" else f"{out_base}, {out_prop}")

