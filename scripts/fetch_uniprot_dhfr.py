import argparse
from pathlib import Path
import time
import requests


def fetch_fasta(query: str, size: int, out_path: Path):
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "fasta",
        "size": str(size),
    }

    # UniProt REST sometimes benefits from retry
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200 and r.text.startswith(">"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(r.text)
            return len(r.text)
        time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch FASTA. status={r.status_code}, text_head={r.text[:200]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of sequences to fetch")
    ap.add_argument("--reviewed", action="store_true", help="only reviewed (Swiss-Prot)")
    ap.add_argument("--bacteria", action="store_true", help="restrict to bacteria (taxonomy_id:2)")
    ap.add_argument("--out", default="data/dhfr_homologs.fasta")
    args = ap.parse_args()

    # Query: DHFR = EC 1.5.1.3 or name contains dihydrofolate reductase
    # Use protein name + EC as a practical filter
    parts = [
        '(protein_name:"dihydrofolate reductase" OR ec:1.5.1.3)',
        "reviewed:true" if args.reviewed else "",
        "taxonomy_id:2" if args.bacteria else "",
    ]
    query = " AND ".join([p for p in parts if p])

    out_path = Path(args.out)
    nbytes = fetch_fasta(query=query, size=args.n, out_path=out_path)
    print("Query:", query)
    print(f"Saved: {out_path} ({nbytes/1024:.1f} KB)")


if __name__ == "__main__":
    main()