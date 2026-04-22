import requests
from typing import List
from xml.etree import ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import json
import time
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# config for extraction
ENTREZ_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_ELINK_URL = f"{ENTREZ_URL}/elink.fcgi"
NCBI_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# GWAS term
# MeSH
GWAS_MESH_TERMS = [
    'Genome-Wide Association Study', 'Genetic Association Studies', 'Polymorphism, Single Nucleotide'
]
GWAS_MESH_FILTER = " OR ".join([term + "[MeSH]" for term in GWAS_MESH_TERMS])
# title and abstract
GWAS_TIAB_TERMS = [
    'GWAS', 'genome-wide association', "polygenic risk", "polygenic score", "genetic variant",
    "genetic risk", "SNP", "single nucleotide polymorphism", "risk loci", "genetic association",
    "meta-analysis", "Mendelian randomization", "whole genome sequencing", "whole exome sequencing",
    "copy number variant", "heritability", "gene expression", "transcriptome", "eQTL",
    "epigenome", "DNA methylation", "Genetic Predisposition to Disease", "Genetic Loci", 
    "Aged", "Humans", "Risk Factors"
]
GWAS_TIAB_FILTER = " OR ".join([term + "[tiab]" for term in GWAS_TIAB_TERMS])
# final filter
GWAS_FILTER = f'({GWAS_MESH_FILTER}) AND ({GWAS_TIAB_FILTER})'

# Alzheimer's Disease term
# MeSH
AD_MESH_TERMS = [
    "Alzheimer Disease/genetics", "Frontotemporal Dementia/genetics", "Parkinson Disease/genetics",
    "Amyotrophic Lateral Sclerosis/genetics", "tau Proteins/genetics", "Apolipoproteins E/genetics",
    "Microtubule-Associated Protein Tau", "Progressive Supranuclear Palsy/genetics",
    "Corticobasal Degeneration/genetics", "Dementia, Vascular/genetics"
]
AD_MESH_FILTER = " OR ".join([term + "[MeSH]" for term in AD_MESH_TERMS])
# title and abstract
AD_TIAB_TERMS = [
    "Alzheimer", "dementia", "cognitive impairment", "neurodegeneration", "Frontotemporal dementia", 
    "Parkinson Disease", "Amyotrophic Lateral Sclerosis", "Progressive Supranuclear Palsy", 
    "Corticobasal Degeneration", "Vascular dementia"
]
AD_TIAB_FILTER = " OR ".join([term + "[tiab]" for term in AD_TIAB_TERMS])
# final filter
AD_FILTER = f'({AD_MESH_FILTER}) AND ({AD_TIAB_FILTER})'

# article filter
ARTICLE_FILTER = '(Journal Article[pt])'
# AND (Meta-Analysis[pt] OR Comparative Study[pt] OR Multicenter Study[pt] OR Evaluation Study[pt] OR Validation Study[pt])

# free text filter
FREE_TEXT_FILTER = '(free full text[sb])'

# known pmids
ADVP1 = pd.read_csv("test_tables/ADVP_1026_v3p8_extracted.txt", sep = "\t", encoding="cp1252")
ADVP1_ALL_PMID = ADVP1["Pubmed ID"].apply(lambda x: str(int(x)) if not pd.isna(x) else "").unique()

# list of mesh terms used
ALL_MESH_TERMS = GWAS_MESH_TERMS + AD_MESH_TERMS

def make_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,          # waits 2, 4, 8, 16, 32s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    return s

def pmids_to_pmcids_map(batch_pmids: List[str]) -> dict:
    """Map PubMed IDs to PMC IDs. Tries elink first, falls back to NCBI ID converter."""
    result = {}
    try:
        r = session.post(
            ENTREZ_ELINK_URL,
            data=[
                ("dbfrom", "pubmed"),
                ("db", "pmc"),
                ("linkname", "pubmed_pmc"),
                ("retmode", "xml"),
                ("api_key", os.environ.get("ENTREZ_API_KEY", "")),
                *[("id", pmid) for pmid in batch_pmids],
            ],
            timeout=60,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        if root.find("ERROR") is not None:
            raise ValueError(f"elink error: {root.findtext('ERROR')}")
        for linkset in root.findall("./LinkSet"):
            pmid_elems = linkset.findall("./IdList/Id")
            if not pmid_elems:
                continue
            pmid = pmid_elems[0].text
            pmcids = []
            for db in linkset.findall("./LinkSetDb"):
                dbto = db.find("DbTo")
                if dbto is not None and dbto.text == "pmc":
                    pmcids.extend([x.text for x in db.findall("./Link/Id")])
            if pmcids:
                result[pmid] = f"PMC{pmcids[0]}"
    except Exception as e:
        print(f"elink failed ({e}), falling back to NCBI ID converter")
        r = session.get(
            NCBI_IDCONV_URL,
            params={
                "ids": ",".join(batch_pmids),
                "format": "json",
                "tool": "gwas_extractor",
                "email": os.environ.get("ENTREZ_EMAIL", ""),
            },
            timeout=60,
        )
        r.raise_for_status()
        for record in r.json().get("records", []):
            if "pmcid" in record and "errmsg" not in record:
                result[str(record["pmid"])] = record["pmcid"]
    return result

def verify_paper_with_gwas_table(pmcid: str) -> bool:
    try:
        has_gwas_table = False
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for d in data:
                doc = d["documents"]
                for p in doc:
                    passage = p["passages"]
                    for item in passage:
                        if item.get("infons", "").get("type", "").lower() == "table" and "text" in item:
                            table_str = item["text"]
                            snp_filter = re.search(r"rs\d+", table_str) is not None
                            chr_filter = (re.search(r"[Cc][Hh][Rr]", table_str) is not None) and (re.search('(?<![A-Za-z.])\d+(?![A-Za-z.])', table_str) is not None)
                            p_value_filter = (re.search(r"\d+\.\d+", table_str) is not None) and (re.search(r"(?<![A-Za-z])[Pp](?![A-Za-z])", table_str) is not None)
                            if (snp_filter or chr_filter) and p_value_filter:
                                has_gwas_table = True
                                break
                    if has_gwas_table:
                        break
                if has_gwas_table:
                    break
        return has_gwas_table
    except Exception as e:
        return False

def extract_papers_by_year(year: int) -> List[dict]:
    # 1. Search for PMIDs
    params = {
        "db": "pubmed",
        "term": f"{GWAS_FILTER} AND {AD_FILTER} AND (Humans[MesH]) AND {FREE_TEXT_FILTER}",
        "datetype": "pdat",
        "mindate": f"{year}/01/01",
        "maxdate": f"{year}/12/31",
        "retmode": "json",
        "retmax": 10000,
        "api_key": os.environ.get("ENTREZ_API_KEY", ""),
    }
    r = session.get(f"{ENTREZ_URL}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    pmids = r.json().get("esearchresult", {}).get("idlist", [])
    pmids_in_advp1 = [
        pmid for pmid in pmids
        if pmid in ADVP1_ALL_PMID
    ]
    if len(pmids_in_advp1) > 0:
        print(f"Found {len(pmids_in_advp1)} papers that was in ADVP1")
    pmids = [
        pmid for pmid in pmids
        if pmid not in ADVP1_ALL_PMID
    ]
    if not pmids:
        return []
    time.sleep(2)
    # 2. Fetch metadata in batches
    papers = []
    for i in range(0, len(pmids), 10):
        batch_pmids = pmids[i:min(i + 10, len(pmids))]
        batch_pmids = [str(pmid) for pmid in batch_pmids]
        # fetch pubmed id -> pmcid
        pmids_to_pmcids = pmids_to_pmcids_map(batch_pmids)

        time.sleep(2)

        removed_pmid = []
        for pmid in pmids_to_pmcids:
            pmcid = pmids_to_pmcids[pmid]
            if not verify_paper_with_gwas_table(pmcid):
                removed_pmid.append(pmid)
            time.sleep(0.33)

        batch_pmids_to_pmcids = list(pmids_to_pmcids.keys())
        batch_pmids_to_pmcids = [str(pmid) for pmid in batch_pmids_to_pmcids if pmid not in removed_pmid]
        if not batch_pmids_to_pmcids:
            time.sleep(2)
            continue
        # fetch other metadata
        r = session.get(
            f"{ENTREZ_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(batch_pmids_to_pmcids),
                "rettype": "xml",
                "retmode": "xml",
                "api_key": os.environ.get("ENTREZ_API_KEY", ""),
            },
            timeout=60,
        )
        r.raise_for_status()

        for article in ET.fromstring(r.content).findall(".//PubmedArticle"):
            art = article.find(".//Article")
            title_el = art.find(".//ArticleTitle")
            # get mesh terms and all qualifiers
            mesh_terms = []
            for mh in article.findall(".//MedlineCitation/MeshHeadingList/MeshHeading"):
                descriptor_el = mh.find("DescriptorName")
                if descriptor_el is None:
                    continue
                descriptor = "".join(descriptor_el.itertext()).strip()
                qualifiers = [
                    "".join(q.itertext()).strip()
                    for q in mh.findall("QualifierName")
                ]
                if qualifiers:
                    for q in qualifiers:
                        mesh_terms.append(f"{descriptor}/{q}")
                mesh_terms.append(descriptor) # also keep this for those that we filter without qualifiers
            # filter for only those in the the filter
            mesh_terms_used = [term for term in mesh_terms if term in ALL_MESH_TERMS]
            try:
                year = int((
                        art.findtext(".//Journal/JournalIssue/PubDate/Year")
                        or art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", "")[:4]
                ))
            except:
                year = None
            papers.append({
                "pmid": article.findtext(".//MedlineCitation/PMID", ""),
                "pmcid": pmids_to_pmcids[article.findtext(".//MedlineCitation/PMID", "")],
                "year": year,
                "title": "".join(title_el.itertext()) if title_el is not None else "",
                "abstract": " ".join(
                    "".join(el.itertext())
                    for el in art.findall(".//AbstractText")
                ),
                "journal": art.findtext(".//Journal/Title", ""),
                "authors": [
                    f"{a.findtext('LastName', '')}, {a.findtext('ForeName', '')}".strip(", ")
                    for a in art.findall(".//AuthorList/Author")
                    if a.findtext("LastName")
                ],
                "MeSH terms used": mesh_terms_used
            })

        time.sleep(2)

    return papers

if __name__ == "__main__":
    session = make_session()
    papers = []
    for year in range(2009, 2027):
        print(year)
        papers.extend(extract_papers_by_year(year))
        print(f"Total paper found until {year}: {len(papers)}")
        print()

    papers = pd.DataFrame(papers)
    papers = papers.sort_values(["year", "pmid"]).reset_index().drop("index", axis = 1)
    papers.to_csv("new_gwas_ad_paper.csv", index = False)