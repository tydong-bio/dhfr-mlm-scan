import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import esm
import matplotlib.pyplot as plt

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def read_fasta_one(path: Path) -> tuple[str, str]:

    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("FASTA must start with a header line beginning with '>'")
    
    seq = "".join(lines[1:]).replace(" ", "").upper()

    bad = sorted(set([c for c in seq if c not in AA20]))
    if bad:
        raise ValueError(f"Sequence contains invalid characters: {bad}")

    return lines[0][1:], seq

@torch.no_grad()
def masked_llr_scores(model, alphabet, seq: str, device: str = "cpu") -> pd.DataFrame:
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [("DHFR", seq)]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)

    # positions in toks: [BOS] + seq + [EOS]
    L = len(seq)
    scores = np.zeros((L, len(AA20)), dtype=np.float32)

    for i in range(L):
        toks_masked = toks.clone()
        toks_masked[0, i + 1] = alphabet.mask_idx
        out = model(toks_masked)
        logits = out["logits"][0, i + 1]  # vocab logits at masked position
        logp = torch.log_softmax(logits, dim=0)

        wt = seq[i]
        wt_idx = alphabet.get_idx(wt)
        wt_logp = float(logp[wt_idx].cpu())

        for j, aa in enumerate(AA20):
            aa_idx = alphabet.get_idx(aa)
            scores[i, j] = float(logp[aa_idx].cpu()) - wt_logp  # LLR vs WT

    df = pd.DataFrame(scores, columns=AA20)
    df.insert(0, "wt", list(seq))
    df.insert(0, "pos", np.arange(1, L + 1))
    return df

def plot_heatmap(df: pd.DataFrame, out_png: Path):
    L = df.shape[0]
    wt = df["wt"].tolist()
    mat = df[AA20].to_numpy().T  # 20 x L

    # 21 x L: 第0行留给 WT 字母，用 NaN 占位（显示为白色）
    full = np.vstack([np.full((1, L), np.nan, dtype=np.float32), mat])

    # 让 NaN 显示为白色
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    # “拉宽每个格子”：每列对应多少英寸（越大越宽）
    col_in = 0.22
    row_in = 0.28
    fig_w = max(12, L * col_in)
    fig_h = (21 * row_in) + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(np.ma.masked_invalid(full), aspect="auto", cmap=cmap)
        # highlight constrained/conserved sites (1-based positions)
    highlight = [43,27,31,1,48,46,81,40,47,50,80,41,60,83,42]
    for p in highlight:
        x = p - 1  # imshow uses 0-based column index
        ax.axvline(x=x, linewidth=0.8)
    # label the most conserved one
    ax.text(43-1, -0.8, "G43", ha="center", va="bottom", fontsize=10, color="black")

    # y 轴：第0行是 WT，其余 20 行是突变氨基酸
    ylabels = ["WT"] + AA20
    ax.set_yticks(np.arange(21))
    ax.set_yticklabels(ylabels)

    # x 轴：每列就是一个位置（严格 1:1）
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(np.arange(1, L + 1), fontsize=6, rotation=90)
    ax.set_xlabel("Position (1..L)")
    ax.set_ylabel("")

    # 在 WT 行每个格子写一个字母（黑字，保证可读）
    for x, aa in enumerate(wt):
        ax.text(x, 0, aa, ha="center", va="center",
                fontsize=8, color="black", family="monospace")

    # 网格线，确保视觉上格子对齐
    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 21, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("DHFR mutation scan (ESM-2 masked LLR)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="log p(mut) - log p(wt)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/dhfr.fasta")
    ap.add_argument("--model", default="esm2_t6_8M_UR50D")
    ap.add_argument("--out_csv", default="results/mutation_scores.csv")
    ap.add_argument("--out_png", default="results/heatmap.png")
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    name, seq = read_fasta_one(fasta)

    model_fn = getattr(esm.pretrained, args.model, None)
    if model_fn is None:
        raise ValueError(f"Unknown model name: {args.model}")

    model, alphabet = model_fn()
    device = "cpu"
    model = model.to(device)

    df = masked_llr_scores(model, alphabet, seq, device=device)
    df.to_csv(out_csv, index=False)
    plot_heatmap(df, out_png)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()