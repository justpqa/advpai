"""
Validate predicted GWAS table terms against term_mapping_dict.json.

For each file in the target directory and for each of the 6 key columns,
parses every cell value, normalises each term, and checks whether it is
a recognised key in the term mapping dict.

Usage:
    python evaluate_harmonization.py [--dir PRED_DIR] [--dict DICT_PATH] [--out RESULTS_PATH]

Defaults:
    --dir   pred_tables
    --dict  term_mapping_dict.json
    --out   term_validation_results.json
"""

import os
import re
import json
import argparse
import unicodedata
import pandas as pd

COLS = ["Population", "Cohort", "Stage", "Imputation", "Study type", "Phenotype"]


# ─────────────────────────── helpers ──────────────────────────────────────────

def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("'", "")
    return re.sub(r"\s+", " ", s.lower().strip())


def _strip_context_prefix(val: str) -> str:
    """Remove prefixes like 'GLOBAL: ', 'IFGC Phase I: '."""
    return re.sub(r"^[^:]+:\s*", "", val).strip()


def parse_cell_terms(raw_value) -> list[str]:
    """
    Split a cell on ' + ', strip context prefixes, and return
    deduplicated candidate strings (both raw and stripped forms).
    """
    if pd.isna(raw_value):
        return []
    terms: set[str] = set()
    for part in str(raw_value).split(" + "):
        part = part.strip()
        if part:
            terms.add(part)
            stripped = _strip_context_prefix(part)
            if stripped:
                terms.add(stripped)
    return list(terms)


def is_term_valid(term: str, col: str, term_map: dict) -> bool:
    """Return True if the normalised term is a recognised key in the dict."""
    return normalise(term) in term_map.get(col, {})


# ─────────────────────────── per-file validation ──────────────────────────────

def validate_file(file_path: str, term_map: dict) -> dict:
    """
    For each of the 6 columns in one prediction file, collect every unique
    term and classify it as valid (in dict) or unknown (not in dict).
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    df.columns = [c.strip() for c in df.columns]

    file_result: dict = {"file": os.path.basename(file_path), "cols": {}}

    for col in COLS:
        if col not in df.columns:
            file_result["cols"][col] = {"note": f"column '{col}' missing"}
            continue

        valid_terms: set[str] = set()
        unknown_terms: set[str] = set()

        for raw_val in df[col].dropna():
            for term in parse_cell_terms(raw_val):
                if is_term_valid(term, col, term_map):
                    valid_terms.add(term)
                else:
                    unknown_terms.add(term)

        total = len(valid_terms) + len(unknown_terms)
        valid_pct = round(len(valid_terms) / total * 100, 1) if total > 0 else None

        file_result["cols"][col] = {
            "valid_count": len(valid_terms),
            "unknown_count": len(unknown_terms),
            "valid_pct": valid_pct,
            "valid_terms": sorted(valid_terms),
            "unknown_terms": sorted(unknown_terms),
        }

    return file_result


# ─────────────────────────── main ─────────────────────────────────────────────

def run(pred_dir: str, dict_path: str, results_path: str) -> list[dict]:
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"term_mapping_dict not found: {dict_path}")

    with open(dict_path) as f:
        raw_map = json.load(f)
    term_map = {col: {normalise(k): v for k, v in mapping.items()} for col, mapping in raw_map.items()}
    print(f"[term_map] Loaded {dict_path}")

    files = sorted(
        f for f in os.listdir(pred_dir)
        if f.endswith(".csv") or f.endswith(".xlsx")
    )
    if not files:
        print(f"No .csv / .xlsx files found in {pred_dir}")
        return []

    all_results: list[dict] = []

    for fname in files:
        fpath = os.path.join(pred_dir, fname)
        result = validate_file(fpath, term_map)
        all_results.append(result)

        print(f"\n{'='*60}")
        print(f"File: {fname}")
        for col in COLS:
            info = result["cols"].get(col, {})
            if "note" in info:
                print(f"  {col:15s}: {info['note']}")
            else:
                pct = info["valid_pct"]
                v, u = info["valid_count"], info["unknown_count"]
                pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
                print(f"  {col:15s}: {pct_str} valid  ({v} valid, {u} unknown)")
                for t in sorted(info["unknown_terms"])[:5]:
                    print(f"    UNKNOWN  '{t}'")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — per-column mean valid %")
    for col in COLS:
        pcts = [
            r["cols"][col]["valid_pct"]
            for r in all_results
            if col in r["cols"] and r["cols"][col].get("valid_pct") is not None
        ]
        if pcts:
            print(f"  {col:15s}: {sum(pcts)/len(pcts):.1f}%  (n={len(pcts)} files)")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pred table terms against term_mapping_dict.json")
    parser.add_argument("--dir",  default="pred_tables",              help="Directory of prediction files")
    parser.add_argument("--dict", default="term_mapping_dict.json",   help="Path to term_mapping_dict.json")
    parser.add_argument("--out",  default="term_validation_results.json", help="Output results JSON path")
    args = parser.parse_args()

    run(args.dir, args.dict, args.out)
