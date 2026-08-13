import os
from copy import deepcopy
import io
from typing import List
import numpy as np
import pandas as pd
import requests
import re
from pypdf import PdfReader, PdfWriter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import DocumentStream
from table_link_to_excel import _try_pmc_direct_table_download, _flatten_columns, sanitize_sheet_name, fetch_html, fetch_pmc_fulltext_xml, pick_table, table_to_dataframe, _clean_text
from bs4 import BeautifulSoup

"""
This files contain all utils functions for extraction and harmonization
"""

def extract_tables_num_col_lst_from_pmc(pmcid: str) -> List[int]:
    """
    Extract the number of col of a table to make sure we try the right orientation with docling
    """

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

    response = requests.get(url)
    tables_num_col_lst = []
    if response.status_code == 200:
        data = response.json()
        for d in data:
            doc = d["documents"]
            for p in doc:
                passage = p["passages"]
                for item in passage:
                    if item.get("infons", "").get("type", "").lower() == "table" and "text" in item:
                        table_str = item["text"]
                        num_col = 0
                        for row in table_str.split("\t \t"):
                            row_lst = row.split("\t")
                            num_col = max(num_col, len(row_lst))
                        tables_num_col_lst.append(num_col)
        print(f"Successfully retrieve number of columns")
    else:
        print(f"Failed to retrieve number of columns: {response.status_code}")

    return tables_num_col_lst

# first we need to extract a set of id from table
def extract_table_id_lst_from_pmc(pmcid: str) -> List[str]:
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

    response = requests.get(url)
    tables_id_set = set()
    if response.status_code == 200:
        data = response.json()
        for d in data:
            doc = d["documents"]
            for p in doc:
                passage = p["passages"]
                for item in passage:
                    if "table" in item.get("infons", "").get("type", "").lower() and "id" in item.get("infons", ""):
                        tables_id_set.add(item.get("infons", "").get("id", ""))
                        
        print(f"Successfully retrieve list of tables' id")
    else:
        print(f"Failed to retrieve list of tables' id {response.status_code}")
        
    return [id for id in list(tables_id_set) if id != ""]

def contains_valid_snp(row):
    row_snp_pattern = r"\b(?:rs|s)\d+\S*"
    return any(re.search(row_snp_pattern, str(value)) for value in row)

def contains_valid_pvalue(row):
    row_pvalue_pattern = r"\d+\.\d+"
    return any(re.search(row_pvalue_pattern, str(value)) for value in row)

def _fetch_pmc_soup(pmcid: str):
    """
    Fetch the article once and return a soup holding every table.

    Ordering matters here. The article HTML page is rate limited: after ~2 requests
    PMC stops serving the article and returns a reCAPTCHA challenge page with
    HTTP 200, which fetch_html treats as a valid body (it only checks resp.ok).
    The full text XML is served from Europe PMC / NCBI OAI / the OA package and is
    stable across runs, so we try it first and keep the HTML page as the fallback.
    """
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed. Run: python3 -m pip install beautifulsoup4")

    try:
        return BeautifulSoup(fetch_pmc_fulltext_xml(pmcid), "xml")
    except Exception as e:
        print(f"Cannot fetch full text XML for {pmcid} with error {e}, falling back to article HTML")

    html = fetch_html(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}")
    # A 200 response is not proof of content: the bot challenge is served as 200.
    if "RecaptchaChallengePageUi" in html or "<table" not in html.lower():
        raise RuntimeError(f"PMC returned a bot challenge page instead of the article for {pmcid}")

    soup = BeautifulSoup(html, "lxml")
    # On the HTML page the <table> itself carries no id; the T1/T2/... id sits on an
    # ancestor wrapper (table < div < section#T1 < section#S15 < ...). Copy the id of
    # the *nearest* id bearing ancestor onto the table so pick_table resolves the same
    # ids the BioC API reports, exactly as it does on the XML. Walking up from each
    # table is what makes this safe: scanning id holders top down would let an outer
    # section (#S15) claim the table before its own wrapper (#T1) is reached.
    for table in soup.find_all("table"):
        if table.get("id"):
            continue
        for parent in table.parents:
            if parent.get("id"):
                table["id"] = parent["id"]
                break
    return soup


def table_link_to_excel(pmid: int, pmcid: str) -> bool:
    # extract list of table id directly, sorted so a partial run is reproducible
    # (extract_table_id_lst_from_pmc builds a set, whose order varies per process)
    table_id_list = sorted(extract_table_id_lst_from_pmc(pmcid))

    os.makedirs("intermediate_tables", exist_ok=True)

    # one fetch per paper, not one per table id, so we stay under the rate limit
    soup = None
    try:
        soup = _fetch_pmc_soup(pmcid)
    except Exception as e:
        print(f"Cannot fetch {pmcid} with error {e}")

    has_error = False
    for table_id in table_id_list:
        table_name = f'intermediate_tables/{pmid}_{pmcid}_{table_id}_from_pmc.xlsx'
        found_table = False
        try:
            if soup is not None:
                try:
                    tables = pick_table(soup, table_id=table_id, table_selector=None, table_index=0)
                except Exception as e:
                    tables = []
                    print(f"Cannot find table {table_id} in the fetched {pmcid} document with error {e}")

                # collect first, so a table that yields nothing leaves no empty file behind
                sheets = []
                for i, table in enumerate(tables, start=1):
                    df = table_to_dataframe(table)
                    if df.empty:
                        continue
                    caption = ""
                    cap = table.find("caption")
                    if cap:
                        caption = _clean_text(cap.get_text(" ", strip=True))
                    df2 = _flatten_columns(df).ffill()
                    sheets.append((sanitize_sheet_name(caption, fallback=f"table_{i}"), df2))

                if sheets:
                    with pd.ExcelWriter(table_name, engine="openpyxl") as writer:
                        for sheet, df2 in sheets:
                            df2.to_excel(writer, index=False, header=False, sheet_name=sheet)
                    found_table = True

            if not found_table:
                # last resort: PMC direct table assets. Kept as a fallback rather than a
                # fast path because it costs 5 requests per table id and its endpoints
                # flap between 404 and 200 on identical URLs.
                direct_tables = [df for df in (_try_pmc_direct_table_download(pmcid, table_id) or []) if not df.empty]
                if direct_tables:
                    with pd.ExcelWriter(table_name, engine="openpyxl") as writer:
                        for i, df in enumerate(direct_tables, start=1):
                            df2 = _flatten_columns(df).ffill()
                            df2.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("", f"table_{i}"))
                    found_table = True

            if not found_table:
                raise ValueError(f"Cannot extract table with id={table_id}")
        except Exception as e:
            print(f"Error in extracting table {table_name} with error {e}")
            has_error = True

        # if found_table:
        #     # filter table
        #     df = pd.read_excel(table_name)
        #     df["valid_row"] = df.apply(lambda x: contains_valid_snp(x) and contains_valid_pvalue(x), axis=1)
        #     df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
        #     if df.shape[0] > 0:
        #         with pd.ExcelWriter(table_name, engine="openpyxl") as writer:
        #             df.to_excel(writer, index=False)
        # else:
        #     # delete file
        #     if table_name in os.listdir("tables"):
        #         os.remove(table_name)
    return has_error

def clean_cell(val):
    """
    Clean any cell string with the tag for extra note (like "3.14 a")
    """
    tag_pattern = r'\s[a-z]$'
    if isinstance(val, str):
        return re.sub(tag_pattern, '', val)
    return val

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Similar to clean_cell, but for col index of a dataframe
    """
    tag_pattern = r'\s[a-z]$'
    new_cols = []
    seen = {}
    for col in df.columns:
        if pd.isna(col):
            new_cols.append("")
        else:
            # 1. Apply the regex to the name string
            clean_name = re.sub(tag_pattern, '', str(col))
            # 2. Handle duplicates (e.g., if 'Price a' and 'Price b' both become 'Price')
            if clean_name in seen:
                seen[clean_name] += 1
                clean_name = f"{clean_name}_{seen[clean_name]}"
            else:
                seen[clean_name] = 0
            new_cols.append(clean_name)
    df.columns = new_cols
    return df

def contains_valid_snp(row):
    row_snp_pattern = r"\b(?:rs|s)\d+\S*"
    return any(re.search(row_snp_pattern, str(value)) for value in row)

def contains_valid_pvalue(row):
    row_pvalue_pattern = r"\d+\.\d+"
    return any(re.search(row_pvalue_pattern, str(value)) for value in row)

def extract_tables_lst_from_pdf_and_num_col(file_name: str, tables_num_col_lst: List[int]) -> List[pd.DataFrame]:
    """
    Extract tables from a paper given a file_name string and a list of number of col for each tables 
    (for cross-check to see if we need to do rotation)
    """
    reader = PdfReader(file_name)
    options = PdfPipelineOptions()
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    ocr_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )
    df_lst = []

    page_num = 1
    while len(df_lst) < len(tables_num_col_lst) and page_num <= len(reader.pages):
        filled = False # flag if we found table
        for angle in [0, 90]: # Try normal, then try rotated
            
            writer = PdfWriter()
            page = reader.pages[page_num - 1]
            
            if angle != 0:
                page.rotate(angle)
            
            writer.add_page(page)
            
            # Convert just this one page
            pdf_buffer = io.BytesIO()
            writer.write(pdf_buffer)
            pdf_buffer.seek(0)
            
            doc_stream = DocumentStream(name=f"page_{page_num}.pdf", stream=pdf_buffer)
            result = ocr_converter.convert(doc_stream)

            # Check if this rotation produced valid table rows
            temp_dfs = []
            for table in result.document.tables:
                df = table.export_to_dataframe()
                # check if table is empty
                if (not df.empty):
                    # Case 1: continue from previous table
                    if len(df_lst) > 0 and df.shape[1] == df_lst[-1].shape[1]:
                        # extra filters for tables that are snp related, we need to remove rows that do not have snp id
                        # often are separation between sections
                        for col in df.columns:
                            # first modify "" -> nan
                            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
                        df["valid_row"] = df.apply(lambda x: contains_valid_snp(x) and contains_valid_pvalue(x), axis=1)
                        df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
                        # some cell have the tag the end (often include a space and a small letter)
                        df = df.map(clean_cell) 
                        df = clean_headers(df) 
                        if df_lst[-1].columns.equals(df.columns):
                            df_lst[-1] = pd.concat([df_lst[-1], df], ignore_index = True)
                            filled = True
                        elif df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]:
                            # fail that test => add to temp since this is a new table
                            temp_dfs.append(df)
                            filled = True
                    # Case 2: new table
                    elif (len(df_lst) + len(temp_dfs)) < len(tables_num_col_lst) and df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]: 
                        # extra filters for tables that are snp related, we need to remove rows that do not have snp id
                        # often are separation between sections
                        for col in df.columns:
                            # first modify "" -> nan
                            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
                        df["valid_row"] = df.apply(lambda x: contains_valid_snp(x) and contains_valid_pvalue(x), axis=1)
                        df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
                        # some cell have the tag the end (often include a space and a small letter)
                        df = df.map(clean_cell) 
                        df = clean_headers(df)             
                        temp_dfs.append(df)
                        filled = True
            if temp_dfs:
                df_lst.extend(temp_dfs)
            
            if filled:
                break
            
        page_num += 1

    # final filter, since we only consider df with snp
    df_lst = [df for df in df_lst if df.shape[0] > 0]
    return df_lst

def extract_tables_lst_from_paper(pmcid: str, file_name: str, table_inx_to_extract: List[int] = []) -> list[pd.DataFrame]:
    """
    Given a paper with pmcid and file name of the pdf file, extract all tables
    """
    tables_num_col_lst = extract_tables_num_col_lst_from_pmc(pmcid)
    if len(table_inx_to_extract) > 0:
        tables_num_col_lst = [tables_num_col_lst[i] for i in table_inx_to_extract]
    df_lst = extract_tables_lst_from_pdf_and_num_col(file_name, tables_num_col_lst)
    return df_lst