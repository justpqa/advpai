import os
import re
import pandas as pd
import json
from collections import Counter
from typing import Iterable, Optional

# Paper tested
test_papers_info = [
    (30448613, "PMC6331247"), (30979435, "PMC6783343"), (28247064, "PMC5613285"),  (30617256, "PMC6836675"),
    (30820047, "PMC6463297"), (29458411, "PMC5819208"), (29777097, "PMC5959890"), (30651383, "PMC6369905"),
    (28780673, "PMC5693762"), (30930738, "PMC6425305"), (31426376, "PMC6723529"), (29967939, "PMC6280657"),
    (29107063, "PMC5920782"), (29274321, "PMC5938137"), (30413934, "PMC6358498"), (30805717, "PMC7193309"),
    (30636644, "PMC6330399"), (29752348, "PMC5976227"), (28560309, "PMC5440281"), (27899424, "PMC5237405"),
]

# extra func to convert to float
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return None if pd.isna(v) else v

        s = str(x).strip()
        if not s:
            return None

        s = (
            s.replace("×", "x")
             .replace("−", "-")
             .replace("–", "-")
             .replace("\u2212", "-")
        ).strip()

        # remove commas and NBSPs: "1,234" or "1 234"
        s = re.sub(r"[,\u00A0]", "", s)

        # normalize "5.2*e-8" / "5.2 * E-8" -> "5.2e-8"
        s = re.sub(r"(?i)\*\s*e\s*([+-]?\s*\d+)\b", r"e\1", s)
        s = re.sub(r"\s+", "", s)  # helps with "3 E -8" and "4.2 x 10 ^ -5"

        # parse "4.2x10^-5" / "4.2*10^-5" (also works with X)
        sci = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:x|\*)10\^?([+-]?\d+)", s, flags=re.I)
        if sci:
            base = float(sci.group(1))
            exp = int(sci.group(2))
            v = base * (10 ** exp)
            return None if pd.isna(v) else v

        # handles "3e-8" and "3E-8" natively
        v = float(s)
        return None if pd.isna(v) else v
    except Exception:
        return None
    
def safe_int(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = int(x)
            if pd.isna(v):
                return None
            return v
        s = str(x).strip()
        s = s.replace(".", "").replace(",", "").replace(" ", "")
        v = int(s)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def create_test_tables_from_advp():
    advp1 = pd.read_csv("../test_tables/advp.variant.records.hg38.tsv", sep = "\t")
    advp1 = advp1.replace("NR", pd.NA)

    # modify column name of those used to test
    advp1 = advp1.rename({
        "Pubmed PMID": "Pubmed ID",
        "Top SNP": "SNP",
        "RA 1(Reported Allele 1)": "RA",
        "OR_nonref": "Effect",
        "#dbSNP_hg38_chr": "Chr",
        "dbSNP_hg38_position": "Position",
        "Cohort_simple3": "Cohort",
        "Population_map": "Population"
    }, axis = 1)[[
        "Pubmed ID", "SNP", "RA", "P-value", "Position",
    ]]

    # Update chr to right format
    # advp1["Chr"] = advp1["Chr"].apply(lambda x: safe_int(x[3:]))
    advp1["P-value"] = advp1["P-value"].apply(lambda x: safe_float(x))
    # advp1["Effect"] = advp1["Effect"].apply(lambda x: safe_float(x))

    for pmid, pmcid in test_papers_info:
        advp1_with_pmid = advp1[advp1["Pubmed ID"] == pmid]
        # sort by snp
        advp1_with_pmid = advp1_with_pmid.sort_values("SNP").reset_index().drop("index", axis = 1)
        advp1_with_pmid.to_csv(f"../test_tables/{pmid}_{pmcid}_old.csv", index = False)

def create_test_tables_from_advp_v2():
    advp1 = pd.read_csv("../test_tables/ADVP_1026_v3p8_extracted.txt", sep = "\t", encoding="cp1252")
    advp1 = advp1.replace("NR", pd.NA)
    # modify column name of those used to test
    advp1 = advp1.rename({
        "Top SNP": "SNP",
        "ReportedAF": "AF",
        "Effect Size (alt vs ref)": "Effect",
        "Effect Size Type (OR or Beta)": "Effect Type",
        "Cohort_simplified_no_counts": "Cohort",
        "LocusName": "Locus",
        "RA 1(Reported Allele 1)": "RA1",
        "RA 2(Reported Allele 2)": "RA2",
        "Table Ref in paper": "Table ID"
    }, axis = 1)[[
        "Pubmed ID", "PMCID", "Table ID", "SNP", "Chr", "RA1", "RA2", "AF", "P-value", "Effect", "Effect Type", "Population", "Cohort", "Stage", "Imputation", "Phenotype", "Study type"
    ]]

    # Update chr to right format
    advp1["Chr"] = advp1["Chr"].apply(lambda x: safe_int(x))
    advp1["P-value"] = advp1["P-value"].apply(lambda x: safe_float(x))
    advp1["Effect"] = advp1["Effect"].apply(lambda x: safe_float(x))
    advp1["AF"] = advp1["AF"].apply(lambda x: safe_float(x))

    for pmid, pmcid in test_papers_info:
        advp1_with_pmid = advp1[advp1["Pubmed ID"] == pmid]
        # sort by snp
        advp1_with_pmid = advp1_with_pmid.sort_values("SNP").reset_index().drop("index", axis = 1)
        advp1_with_pmid.to_csv(f"../test_tables/{pmid}_{pmcid}.csv", index = False)
    
def create_test_tables_from_advp_v3():
    # NOTE: unstable version of tables for testing, skip for now
    advp1_1 = pd.read_csv("../test_tables/advp.variant.records.hg38.tsv", sep = "\t")
    advp1_1 = advp1_1.replace("NR", pd.NA)
    advp1_1 = advp1_1.rename({
        "Pubmed PMID": "Pubmed ID",
        "Top SNP": "SNP",
        "RA 1(Reported Allele 1)": "RA",
        "OR_nonref": "Effect",
        "#dbSNP_hg38_chr": "Chr",
        "dbSNP_hg38_position": "Position",
        "Cohort_simple3": "Cohort",
        "Population_map": "Population"
    }, axis = 1)[[
        "Pubmed ID", "SNP", "RA", "P-value", "Position",
    ]]
    advp1_1["P-value"] = advp1_1["P-value"].apply(lambda x: safe_float(x))
    advp1_1["row_num"] = advp1_1.sort_values(["SNP", "P-value"]).groupby("SNP").cumcount() + 1

    advp1_2 = pd.read_csv("../test_tables/ADVP_1026_v3p8_extracted.txt", sep = "\t", encoding="cp1252")
    advp1_2 = advp1_2.replace("NR", pd.NA)
    # modify column name of those used to test
    advp1_2 = advp1_2.rename({
        "Top SNP": "SNP",
        "ReportedAF": "AF",
        "Effect Size (alt vs ref)": "Effect",
        "Effect Size Type (OR or Beta)": "Effect Type",
        "Cohort_simplified_no_counts": "Cohort",
        "LocusName": "Locus"
    }, axis = 1)[[
        "Pubmed ID", "PMCID", "SNP", "Chr", "Locus", "AF", "P-value", "Effect", "Effect Type", "Population", "Cohort", "Stage",
    ]]

    # Update chr to right format
    advp1_2["Chr"] = advp1_2["Chr"].apply(lambda x: safe_int(x))
    advp1_2["P-value"] = advp1_2["P-value"].apply(lambda x: safe_float(x))
    advp1_2["Effect"] = advp1_2["Effect"].apply(lambda x: safe_float(x))
    advp1_2["AF"] = advp1_2["AF"].apply(lambda x: safe_float(x))
    advp1_2["row_num"] = advp1_2.sort_values(["SNP", "P-value"]).groupby("SNP").cumcount() + 1

    for pmid, pmcid in test_papers_info:
        advp1_with_pmid_1 = advp1_1[advp1_1["Pubmed ID"] == pmid].reset_index().drop("index", axis = 1)
        advp1_with_pmid_2 = advp1_2[advp1_2["Pubmed ID"] == pmid].reset_index().drop("index", axis = 1)
        # advp1_with_pmid_1.to_csv(f"test_tables/{pmid}_{pmcid}_1.csv", index = False)
        # advp1_with_pmid_2.to_csv(f"test_tables/{pmid}_{pmcid}_2.csv", index = False)
        advp1_with_pmid = advp1_with_pmid_1.merge(advp1_with_pmid_2, how = "inner", on = ["Pubmed ID", "SNP", "P-value", "row_num"])
        # sort by snp
        advp1_with_pmid = advp1_with_pmid.sort_values("SNP").reset_index().drop("index", axis = 1)
        advp1_with_pmid.to_csv(f"../test_tables/{pmid}_{pmcid}.csv", index = False)


if __name__ == "__main__":
    create_test_tables_from_advp_v2()