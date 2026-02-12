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
    mat = df[AA20].to_numpy().T  # 20 x L
    fig, ax = plt.subplots(figsize=(max(10, df.shape[0] / 8), 6))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(np.arange(len(AA20)))
    ax.set_yticklabels(AA20)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mutant AA")
    ax.set_title("ESM-2 masked LLR mutation scan (higher = more plausible)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="log p(mut) - log p(wt)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
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