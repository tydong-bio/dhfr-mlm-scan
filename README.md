# dhfr-mlm-scan

This repo is a small, reproducible demo of using a protein language model (ESM-2) to score single-amino-acid substitutions in E. coli DHFR and relate those scores to evolutionary conservation.

# What it does
	•	**Mutation scan (ESM-2 MLM scoring):** for each DHFR position, computes a log-likelihood ratio (LLR) for mutating to each of the 20 amino acids, and visualizes the full 20×L landscape as a heatmap (1:1 aligned with the WT sequence).
	•	**Conservation validation (MSA-based)**: downloads DHFR homologs from UniProt, builds a multiple sequence alignment with MAFFT, computes per-position conservation (Shannon entropy / max frequency), and compares conservation vs ESM “mutability” (e.g., highlighting highly constrained sites like G43).

# What you can use it for
	•	Quickly identify positions that are likely constrained vs tolerant to mutation (hypothesis generation for experiments).
	•	Cross-check model predictions against evolutionary conservation (sanity check / prioritization).
	•	A clean starting template for “protein LLM → variant scoring → plots → GitHub-ready pipeline”.

# ESM-2 mutation scan
python scripts/scan.py --model esm2_t6_8M_UR50D

# Rank most constrained sites (from ESM scores)
python scripts/rank_sites.py --top 15

# Fetch DHFR homologs (UniProt) and run conservation vs ESM comparison
python scripts/fetch_uniprot_dhfr.py --n 500 --bacteria
mafft --auto data/dhfr_plus_homologs.fasta > results/msas/dhfr_plus_msa.fasta
python scripts/conservation_stream.py
