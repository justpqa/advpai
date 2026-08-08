# ============================================================
# Focused AD genetics PubMed retrieval — CADRO categories D, A, B
# ============================================================
# Requirements:
#   pip install biopython pandas openpyxl
#
# Outputs (written to same directory as this script):
#   1. raw_fetch_all_papers.csv       — every paper retrieved from PubMed
#                                       before any post-fetch filtering,
#                                       with PMCID, MeSH terms, publication
#                                       type, and drop_reason column
#   2. new_ad_papers_enriched_v2.csv  — papers that passed all filters,
#                                       with CADRO labels and fetch_reason
# ============================================================

from Bio import Entrez
import pandas as pd
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

# =========================
# CONFIG — edit these
# =========================
Entrez.email   = "your.email@institution.edu"   # <-- CHANGE THIS
INPUT_CSV      = "advp_papers_final.csv"
CADRO_JSON     = "cadro_rules_detailed.json"
RAW_CSV        = "raw_fetch_all_papers.csv"
OUTPUT_CSV     = "new_ad_papers_enriched_v2.csv"

# Post-fetch thresholds
MIN_CADRO_HITS    = 2   # keyword hits in title+abstract per CADRO category
MIN_GENETICS_HITS = 3   # distinct genetics-pattern hits in title+abstract

# =========================
# GENETICS EVIDENCE PATTERNS
# =========================
GENETICS_PATTERNS = [
    r"\bvariant[s]?\b", r"\ballele[s]?\b", r"\bSNP[s]?\b",
    r"\bpolymorphism[s]?\b", r"\bgenotype[s]?\b",
    r"\bgenome.wide\b", r"\bGWAS\b",
    r"\bpolygenic\b", r"\bPRS\b",
    r"\bhaplotype[s]?\b", r"\blocus\b", r"\bloci\b",
    r"\brisk loci\b", r"\bheritability\b",
    r"\bmendelian randomization\b",
    r"\bmeta.analysis\b",
    r"\bp.value[s]?\b", r"\bOR\b.*\bCI\b",
    r"\bβ\b.*\bSE\b",
    r"\b[A-Z][A-Z0-9]{1,8}\d\b",
    r"\brs\d{5,}\b",
    r"\b[Pp]\s*[<=>]\s*\d+\.\d+[eE]?-?\d*",
    r"\bexome\b", r"\bwhole.genome sequencing\b",
    r"\bsequencing\b", r"\bmutation[s]?\b",
    r"\bcopy number\b", r"\bCNV[s]?\b",
    r"\bepigenom\w*\b", r"\bmethylation\b",
    r"\btranscriptom\w*\b", r"\beQTL[s]?\b",
    r"\bsingle.cell\b",
]

# =========================
# AD / RELATED DISORDER FILTER
# =========================
AD_TERMS = re.compile(
    r"alzheimer|amyloid.beta|a[bβ]42|a[bβ]40|tau|tauopathy|"
    r"APOE|apolipoprotein E|late.onset AD|early.onset AD|LOAD|EOAD|"
    r"frontotemporal dementia|FTD|Lewy body|vascular dementia|"
    r"mixed dementia|mild cognitive impairment|MCI|"
    r"cognitive decline|neurodegeneration|dementia|memory loss|brain aging",
    re.IGNORECASE
)

# Publication types to EXCLUDE
# Note: "Meta-Analysis" here means the article IS a meta-analysis type
# (i.e. a review-style synthesis). Remove from set to keep meta-analysis articles.
EXCLUDE_PUB_TYPES = {
    "Review", "Systematic Review",
    "Editorial", "Letter", "Comment", "News",
    "Published Erratum", "Preprint",
    "Retracted Publication",
}

# Preprint journals / servers (catches any not flagged by pub-type field)
PREPRINT_JOURNALS = re.compile(
    r"biorxiv|medrxiv|preprint|ssrn|chemrxiv|researchsquare",
    re.IGNORECASE
)

# =========================
# CADRO QUERY BLOCKS
# =========================
AD_CORE = (
    '("Alzheimer Disease"[mh] OR "Alzheimer"[tiab] OR '
    '"dementia"[tiab] OR "cognitive impairment"[tiab] OR '
    '"neurodegeneration"[tiab]) '
    'NOT review[pt] '
    'AND ("2020/01/01"[pdat] : "3000"[pdat])'
)

GENETICS_CORE = (
    '(GWAS[tiab] OR "genome-wide association"[tiab] OR '
    '"polygenic risk"[tiab] OR "polygenic score"[tiab] OR '
    '"genetic variant"[tiab] OR "genetic risk"[tiab] OR '
    '"SNP"[tiab] OR "single nucleotide polymorphism"[tiab] OR '
    '"risk loci"[tiab] OR "genetic association"[tiab] OR '
    '"meta-analysis"[tiab] OR "Mendelian randomization"[tiab] OR '
    '"whole genome sequencing"[tiab] OR "whole exome sequencing"[tiab] OR '
    '"copy number variant"[tiab] OR "heritability"[tiab] OR '
    '"gene expression"[tiab] OR "transcriptome"[tiab] OR '
    '"eQTL"[tiab] OR "epigenome"[tiab] OR "DNA methylation"[tiab])'
)

CADRO_REFINEMENTS = {
    "D": (
        '(GWAS[tiab] OR "genome-wide association"[tiab] OR '
        '"polygenic risk score"[tiab] OR "PRS"[tiab] OR '
        '"meta-analysis"[tiab] OR "genetic architecture"[tiab] OR '
        '"risk loci"[tiab] OR "Mendelian randomization"[tiab] OR '
        '"population-based"[tiab] OR "cohort"[tiab] OR '
        '"heritability"[tiab] OR "genetic epidemiology"[tiab] OR '
        '"common variant"[tiab] OR "rare variant"[tiab])'
    ),
    "A": (
        '(amyloid[tiab] OR "tau"[tiab] OR "neuroinflammation"[tiab] OR '
        '"microglia"[tiab] OR "synaptic"[tiab] OR "neurodegeneration"[tiab] OR '
        '"mitochondria"[tiab] OR "oxidative stress"[tiab] OR '
        '"protein aggregation"[tiab] OR "gene expression"[tiab] OR '
        '"transcriptome"[tiab] OR "proteome"[tiab] OR '
        '"epigenetic"[tiab] OR "methylation"[tiab] OR '
        '"functional genomics"[tiab] OR "systems biology"[tiab] OR '
        '"eQTL"[tiab] OR "single-cell"[tiab])'
    ),
    "B": (
        '(biomarker[tiab] OR "CSF"[tiab] OR "plasma"[tiab] OR '
        '"PET"[tiab] OR "MRI"[tiab] OR "imaging"[tiab] OR '
        '"amyloid PET"[tiab] OR "tau PET"[tiab] OR '
        '"endophenotype"[tiab] OR "quantitative trait"[tiab] OR '
        '"digital biomarker"[tiab] OR "fluid biomarker"[tiab] OR '
        '"early detection"[tiab] OR "risk prediction"[tiab])'
    ),
}

# =========================
# CADRO SCORING HELPERS
# =========================
def load_cadro_rules(path):
    with open(path) as f:
        data = json.load(f)
    rules = {}
    for cat in data["categories"]:
        rules[cat["id"]] = {
            "keywords": [k.lower() for k in cat.get("keywords", [])],
            "phrases":  [p.lower() for p in cat.get("phrases",  [])],
        }
    return rules


def score_text_against_cadro(text, rules):
    text_lc = text.lower()
    scores = {}
    for cat_id, rule in rules.items():
        hits = []
        for kw in rule["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lc):
                hits.append(kw)
        for ph in rule["phrases"]:
            if ph in text_lc:
                hits.append(ph)
        scores[cat_id] = (len(hits), hits)
    return scores


def cadro_labels(scores, top_cats, min_hits=MIN_CADRO_HITS):
    qualifying = {
        cat: (n, terms) for cat, (n, terms) in scores.items()
        if cat in top_cats and n >= min_hits
    }
    if not qualifying:
        return None, None, None
    sorted_cats = sorted(qualifying.items(), key=lambda x: -x[1][0])
    primary    = sorted_cats[0][0]
    multi_tags = ";".join(c for c, _ in sorted_cats)
    reason     = " | ".join(
        f"{c}({n} hits: {', '.join(t[:6])})"
        for c, (n, t) in sorted_cats
    )
    return multi_tags, primary, reason


# =========================
# GENETICS EVIDENCE CHECK
# =========================
def count_genetics_hits(text):
    return sum(
        1 for pat in GENETICS_PATTERNS
        if re.search(pat, text, re.IGNORECASE)
    )


# =========================
# PUBMED FETCH FUNCTIONS
# =========================
def search_pubmed_all(query, retmax=50000):
    """Return all PMIDs for a query using Entrez history server."""
    handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
    res = Entrez.read(handle); handle.close()
    total     = int(res["Count"])
    web_env   = res["WebEnv"]
    query_key = res["QueryKey"]
    print(f"    Total hits: {total}")

    pmids = []
    for start in range(0, min(total, retmax), 10000):
        handle = Entrez.esearch(
            db="pubmed", term=query, usehistory="y",
            WebEnv=web_env, query_key=query_key,
            retstart=start, retmax=10000,
        )
        r = Entrez.read(handle); handle.close()
        pmids.extend(r["IdList"])
        time.sleep(0.4)
    return pmids


def fetch_pubmed_details(pmids):
    """
    Fetch full metadata from PubMed XML including:
      title, journal, year, authors, abstract,
      publication types, MeSH descriptor names.
    """
    records = []
    total = len(pmids)
    for i in range(0, total, 100):
        batch = pmids[i:i+100]
        if i % 1000 == 0:
            print(f"  Fetching metadata {i}/{total}...")
        try:
            handle   = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
            xml_data = handle.read()
            handle.close()
            root     = ET.fromstring(xml_data)
        except Exception as e:
            print(f"  Fetch error at batch {i}: {e}")
            time.sleep(2)
            continue

        for article in root.findall(".//PubmedArticle"):
            pmid    = article.findtext(".//PMID") or ""
            title   = article.findtext(".//ArticleTitle") or ""
            journal = article.findtext(".//Journal/Title") or ""
            year    = (
                article.findtext(".//PubDate/Year") or
                article.findtext(".//PubDate/MedlineDate") or ""
            )[:4]

            # Authors
            authors = []
            for a in article.findall(".//Author"):
                last  = a.findtext("LastName")
                first = a.findtext("ForeName")
                if last:
                    authors.append(f"{first} {last}" if first else last)

            # Abstract — handle structured (labelled) abstracts
            abs_parts = []
            for el in article.findall(".//AbstractText"):
                label = el.get("Label")
                text  = el.text or ""
                abs_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abs_parts).strip()

            # Publication types
            pub_types = [
                pt.text for pt in article.findall(".//PublicationType") if pt.text
            ]

            # MeSH descriptor names (all headings, major and minor)
            mesh_terms = []
            for mh in article.findall(".//MeshHeading"):
                desc = mh.findtext("DescriptorName")
                if desc:
                    mesh_terms.append(desc)

            records.append({
                "pmid":       pmid,
                "title":      title,
                "journal":    journal,
                "year":       year,
                "authors":    "; ".join(authors),
                "abstract":   abstract,
                "pub_types":  "; ".join(pub_types),
                "mesh_terms": "; ".join(mesh_terms),
            })

        time.sleep(0.4)

    return pd.DataFrame(records)


def fetch_pmcids(pmids):
    """Return dict pmid -> 'PMCxxxxxxx' (or None) for a list of PMIDs."""
    mapping = {}
    total = len(pmids)
    for i in range(0, total, 200):
        batch = pmids[i:i+200]
        if i % 2000 == 0:
            print(f"  Fetching PMCIDs {i}/{total}...")
        try:
            handle  = Entrez.elink(
                dbfrom="pubmed", db="pmc",
                id=",".join(batch), linkname="pubmed_pmc"
            )
            records = Entrez.read(handle); handle.close()
            for rec in records:
                if not rec["IdList"]:
                    continue
                pmid  = rec["IdList"][0]
                links = rec.get("LinkSetDb", [])
                mapping[pmid] = ("PMC" + links[0]["Link"][0]["Id"]) if links else None
        except Exception as e:
            print(f"  PMC link error at batch {i}: {e}")
        time.sleep(0.4)
    return mapping


# =========================
# MAIN
# =========================
def main():
    # ---- Load inputs ----
    print("Loading ADVP data...")
    df_advp     = pd.read_csv(INPUT_CSV)
    cadro_rules = load_cadro_rules(CADRO_JSON)

    # ---- Top 3 CADRO categories (by frequency, excluding uncategorized) ----
    top_cats = [
        c for c in df_advp["cadro_primary"].value_counts().index
        if c != "uncategorized"
    ][:3]
    print(f"Top 3 CADRO categories: {top_cats}")

    # ---- Search PubMed per category ----
    pmid_cats = defaultdict(set)   # pmid -> set of CADRO categories that found it
    for cat in top_cats:
        refinement = CADRO_REFINEMENTS.get(cat)
        if not refinement:
            print(f"  No refinement for category {cat}, skipping.")
            continue
        query = f"{AD_CORE} AND {GENETICS_CORE} AND {refinement}"
        print(f"\nSearching CADRO category {cat}...")
        pmids = search_pubmed_all(query)
        for p in pmids:
            pmid_cats[p].add(cat)
        time.sleep(1)

    all_pmids = list(pmid_cats.keys())
    print(f"\nTotal unique PMIDs retrieved: {len(all_pmids)}")

    # ---- Remove papers already in ADVP ----
    advp_pmids = set(df_advp["pmid"].astype(str))
    new_pmids  = [p for p in all_pmids if p not in advp_pmids]
    advp_dupes = len(all_pmids) - len(new_pmids)
    print(f"Already in ADVP (excluded): {advp_dupes}")
    print(f"New PMIDs to process:       {len(new_pmids)}")

    # ---- Fetch full metadata for ALL new papers ----
    print("\nFetching PubMed metadata for all new papers...")
    meta_df = fetch_pubmed_details(new_pmids)
    print(f"Metadata fetched for {len(meta_df)} papers")

    # ---- Fetch PMCIDs for ALL new papers ----
    print("\nFetching PMCIDs for all new papers...")
    pmcid_map = fetch_pmcids(new_pmids)
    meta_df["pmcid"] = meta_df["pmid"].map(lambda x: pmcid_map.get(x))

    # ---- Tag each paper with which CADRO search categories found it ----
    meta_df["cadro_search_cats"] = meta_df["pmid"].map(
        lambda x: ";".join(sorted(pmid_cats.get(x, [])))
    )

    # ============================================================
    # POST-FETCH FILTERING
    # Every paper is written to the RAW sheet regardless of outcome.
    # drop_reason is populated with all reasons it was excluded,
    # or "KEPT" if it passed every filter.
    # ============================================================
    print("\nApplying post-fetch filters...")

    counts = {
        "review_preprint": 0,
        "non_ad":          0,
        "no_genetics":     0,
        "no_cadro":        0,
        "kept":            0,
    }

    raw_rows  = []
    kept_rows = []

    for _, row in meta_df.iterrows():
        text      = f"{row['title']} {row['abstract']}"
        pub_types = row.get("pub_types", "") or ""
        journal   = row.get("journal",   "") or ""
        reasons   = []

        # ---- Filter 1: review or preprint ----
        pt_set         = {pt.strip() for pt in pub_types.split(";") if pt.strip()}
        flagged_pts    = sorted(pt_set & EXCLUDE_PUB_TYPES)
        is_preprint_jl = bool(PREPRINT_JOURNALS.search(journal))
        if flagged_pts or is_preprint_jl:
            counts["review_preprint"] += 1
            detail = (
                f"Review/preprint: pub_type flags=[{', '.join(flagged_pts) or 'none'}]"
                + (f"; journal='{journal}' matches preprint pattern" if is_preprint_jl else "")
            )
            reasons.append(detail)

        # ---- Filter 2: must be about AD / related disorders ----
        if not AD_TERMS.search(text):
            counts["non_ad"] += 1
            reasons.append(
                "Non-AD: no Alzheimer/dementia/MCI/neurodegeneration terms "
                "found in title+abstract"
            )

        # ---- Filter 3: genetics evidence in text ----
        gen_hits = count_genetics_hits(text)
        if gen_hits < MIN_GENETICS_HITS:
            counts["no_genetics"] += 1
            reasons.append(
                f"Weak genetics evidence: {gen_hits} of {MIN_GENETICS_HITS} "
                f"required pattern(s) matched in title+abstract"
            )

        # ---- Filter 4: CADRO keyword scoring ----
        cadro_scores = score_text_against_cadro(text, cadro_rules)
        multi_tags, primary, cadro_reason = cadro_labels(cadro_scores, set(top_cats))
        if primary is None:
            counts["no_cadro"] += 1
            reasons.append(
                f"No CADRO match: keyword hits < {MIN_CADRO_HITS} "
                f"for all of {top_cats} in title+abstract"
            )

        drop_str = " | ".join(reasons) if reasons else "KEPT"

        # ---- RAW row (written for every paper) ----
        raw_rows.append({
            "pmid":              row["pmid"],
            "title":             row["title"],
            "journal":           row["journal"],
            "year":              row["year"],
            "authors":           row["authors"],
            "abstract":          row["abstract"],
            "pub_types":         row["pub_types"],
            "mesh_terms":        row["mesh_terms"],
            "pmcid":             row["pmcid"],
            "cadro_search_cats": row["cadro_search_cats"],
            "drop_reason":       drop_str,
        })

        # ---- KEPT row ----
        if not reasons:
            counts["kept"] += 1
            fetch_reason = (
                f"PubMed query match for CADRO {row['cadro_search_cats']}; "
                f"AD/neurodegeneration confirmed in title+abstract; "
                f"genetics evidence hits: {gen_hits}; "
                f"CADRO scoring: {cadro_reason}"
            )
            kept_rows.append({
                "pmid":              row["pmid"],
                "title":             row["title"],
                "journal":           row["journal"],
                "year":              row["year"],
                "authors":           row["authors"],
                "abstract":          row["abstract"],
                "pub_types":         row["pub_types"],
                "mesh_terms":        row["mesh_terms"],
                "pmcid":             row["pmcid"],
                "cadro_search_cats": row["cadro_search_cats"],
                "cadro_multi_tags":  multi_tags,
                "cadro_primary":     primary,
                "cadro_reason":      cadro_reason,
                "fetch_reason":      fetch_reason,
            })

    # ---- Summary ----
    print(f"\n--- Filter summary ---")
    print(f"  Excluded (review/preprint):       {counts['review_preprint']}")
    print(f"  Excluded (non-AD):                {counts['non_ad']}")
    print(f"  Excluded (insufficient genetics): {counts['no_genetics']}")
    print(f"  Excluded (no CADRO match):        {counts['no_cadro']}")
    print(f"  KEPT:                             {counts['kept']}")
    print(f"  (a paper may be excluded for multiple reasons simultaneously)")

    # ---- Save RAW sheet ----
    raw_df = pd.DataFrame(raw_rows, columns=[
        "pmid", "title", "journal", "year", "authors", "abstract",
        "pub_types", "mesh_terms", "pmcid", "cadro_search_cats", "drop_reason",
    ])
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"\nRaw pre-filter sheet → {RAW_CSV}  ({len(raw_df)} papers total)")

    # ---- Save KEPT (filtered) sheet ----
    out_df = pd.DataFrame(kept_rows, columns=[
        "pmid", "title", "journal", "year", "authors", "abstract",
        "pub_types", "mesh_terms", "pmcid", "cadro_search_cats",
        "cadro_multi_tags", "cadro_primary", "cadro_reason", "fetch_reason",
    ])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Filtered output sheet → {OUTPUT_CSV}  ({len(out_df)} papers kept)")

    if not out_df.empty:
        print("\nSample of kept papers:")
        print(
            out_df[["pmid", "year", "cadro_multi_tags", "cadro_primary"]]
            .head(10).to_string()
        )


if __name__ == "__main__":
    main()