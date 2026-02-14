import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="results/mutation_scores.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out_csv", default="results/site_ranking.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # only the 20 mutant columns
    M = df[AA20].to_numpy(dtype=float)  # shape: L x 20
    # exclude the "mut == wt" entry (it is always 0 and inflates best/mean)
    wt = df["wt"].to_numpy()
    for i, aa in enumerate(wt):
        j = AA20.index(aa)
        M[i, j] = np.nan  # ignore wt->wt
    mean_llr = np.nanmean(M, axis=1)
    best_llr = np.nanmax(M, axis=1)
    worst_llr = np.nanmin(M, axis=1)
    frac_ok = np.nanmean(M > -1.0, axis=1)

    out = pd.DataFrame({
        "pos": df["pos"].to_numpy(),
        "wt": df["wt"].to_numpy(),
        "mean_llr": mean_llr,
        "best_llr": best_llr,
        "worst_llr": worst_llr,
        "frac_ok_gt_-1": frac_ok,
    })

    # rank: most constrained = lowest mean_llr (tie-break by lowest best_llr)
    out = out.sort_values(["mean_llr", "best_llr"], ascending=[True, True]).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    print("\nTop most constrained sites:")
    print(out.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()