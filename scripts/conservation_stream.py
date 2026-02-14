import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}

def fasta_iter(path: Path):
    name = None
    seq_chunks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks)
                name = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            yield name, "".join(seq_chunks)

def shannon_entropy(counts_20: np.ndarray) -> float:
    total = counts_20.sum()
    if total <= 0:
        return np.nan
    p = counts_20 / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa", default="results/msas/dhfr_plus_msa.fasta")
    ap.add_argument("--esm_csv", default="results/mutation_scores.csv")
    ap.add_argument("--out_csv", default="results/conservation.csv")
    ap.add_argument("--out_png", default="results/conservation_vs_esm.png")
    ap.add_argument("--highlight", default="43,27,31,1,48,46,81,40,47,50,80,41,60,83,42")
    args = ap.parse_args()

    msa_path = Path(args.msa)

    it = fasta_iter(msa_path)
    dhfr_name, dhfr_aln = next(it)  # first record should be DHFR
    if "DHFR" not in dhfr_name.upper():
        print(f"Warning: first record is {dhfr_name!r}, expected DHFR (still proceeding).")

    L = len(dhfr_aln)
    # map MSA column -> DHFR position (1..), 0 if DHFR has gap at that column
    col_to_pos = np.zeros(L, dtype=np.int32)
    pos_to_col = []
    pos = 0
    for c, ch in enumerate(dhfr_aln):
        if ch != "-":
            pos += 1
            col_to_pos[c] = pos
            pos_to_col.append(c)
    dhfr_len = pos

    # counts per MSA column (only 20 AA; gaps counted separately)
    counts = np.zeros((L, 20), dtype=np.uint16)
    gap_counts = np.zeros(L, dtype=np.uint16)
    nseq = 1  # include DHFR itself in gap counting, but not in AA counts (optional)

    def add_sequence(seq: str):
        nonlocal nseq
        if len(seq) != L:
            return  # skip weird lengths
        nseq += 1
        for c, ch in enumerate(seq):
            if ch == "-":
                gap_counts[c] += 1
            else:
                j = AA_TO_IDX.get(ch)
                if j is not None:
                    counts[c, j] += 1

    # add remaining sequences
    for name, seq in it:
        add_sequence(seq)

    # compute conservation per DHFR position (ignore columns where DHFR has gap)
    entropy = np.full(dhfr_len, np.nan, dtype=float)
    max_freq = np.full(dhfr_len, np.nan, dtype=float)
    gap_frac = np.full(dhfr_len, np.nan, dtype=float)

    for p, c in enumerate(pos_to_col, start=1):
        aa_counts = counts[c].astype(float)
        ent = shannon_entropy(aa_counts)
        entropy[p - 1] = ent
        tot = aa_counts.sum()
        max_freq[p - 1] = float(aa_counts.max() / tot) if tot > 0 else np.nan
        gap_frac[p - 1] = float(gap_counts[c] / nseq)

    cons = pd.DataFrame({
        "pos": np.arange(1, dhfr_len + 1),
        "entropy": entropy,          # lower => more conserved
        "max_freq": max_freq,        # higher => more conserved
        "gap_frac": gap_frac,
    })

    # ESM per-position mean LLR (exclude mut==wt)
    esm = pd.read_csv(args.esm_csv)
    M = esm[AA20].to_numpy(dtype=float)
    wt = esm["wt"].to_numpy()
    for i, aa in enumerate(wt):
        if aa in AA_TO_IDX:
            M[i, AA_TO_IDX[aa]] = np.nan
    esm_mean_llr = np.nanmean(M, axis=1)
    esm_tbl = pd.DataFrame({"pos": esm["pos"].to_numpy(), "wt": esm["wt"].to_numpy(), "esm_mean_llr": esm_mean_llr})

    merged = cons.merge(esm_tbl, on="pos", how="inner")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    # plot
    highlight = [int(x) for x in args.highlight.split(",") if x.strip()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged["entropy"], merged["esm_mean_llr"], s=10, alpha=0.4)
    sel = merged[merged["pos"].isin(highlight)]
    ax.scatter(sel["entropy"], sel["esm_mean_llr"], s=60, alpha=0.9)

    ax.set_xlabel("MSA Shannon entropy (lower = more conserved)")
    ax.set_ylabel("ESM mean LLR (lower = less mutable)")
    ax.set_title("DHFR conservation vs ESM mutability")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=250)
    plt.close(fig)

    print(f"MSA sequences processed: {nseq}")
    print(f"DHFR aligned length (no gaps): {dhfr_len}")
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_png}")

if __name__ == "__main__":
    main()