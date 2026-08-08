import argparse
import io
import re
import subprocess
import tarfile
from typing import List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def fetch_html(url: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python3 -m pip install requests")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    resp = requests.get(url, timeout=timeout, headers=headers)
    if resp.ok and resp.text:
        return resp.text
    # Fallback: some PMC pages block Python requests but allow curl.
    if resp.status_code == 403:
        curl_text = _curl_get_text(url, timeout=timeout)
        if curl_text:
            return curl_text
    resp.raise_for_status()
    return resp.text


def _curl_get_text(url: str, timeout: int = 60) -> Optional[str]:
    try:
        cmd = [
            "curl",
            "-L",
            "--max-time",
            str(timeout),
            "-A",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "-H",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout
    except Exception:
        pass
    return None


def _extract_pmc_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"/articles/(PMC\d+)(?:/table/([^/?#]+)/?)?", url, flags=re.I)
    if not m:
        return None, None
    pmcid = m.group(1).upper()
    table_id = m.group(2)
    return pmcid, table_id


def _try_pmc_direct_table_download(pmcid: str, table_id: str, timeout: int = 60) -> Optional[List[pd.DataFrame]]:
    if requests is None:
        return None
    pmcid_num = re.sub(r"^PMC", "", pmcid, flags=re.I)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    candidates = [
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.csv",
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.htm",
        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{pmcid_num}/bin/{table_id}.html",
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/{table_id}/?download=1",
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/{table_id}/?download=csv",
    ]

    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            ctype = (r.headers.get("Content-Type") or "").lower()
            text = r.text if (r.ok and r.text.strip()) else ""
            if (not text) and r.status_code == 403:
                text = _curl_get_text(url, timeout=timeout) or ""
                ctype = "text/html"
            if not text:
                continue
            if "csv" in ctype or re.search(r",", text.splitlines()[0] if text.splitlines() else ""):
                try:
                    return [pd.read_csv(io.StringIO(text))]
                except Exception:
                    pass
            # try parse as HTML table(s)
            try:
                dfs = pd.read_html(io.StringIO(text))
                dfs = [d for d in dfs if not d.empty]
                if dfs:
                    return dfs
            except Exception:
                pass
        except Exception:
            continue
    return None


def fetch_pmc_fulltext_xml(pmcid: str, timeout: int = 60) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python3 -m pip install requests")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/xml,text/xml,*/*"}

    errors = []

    # 1) Europe PMC full text XML
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        resp = requests.get(api_url, timeout=timeout, headers=headers)
        if resp.ok and resp.text.strip():
            return resp.text
        if resp.status_code == 403:
            curl_text = _curl_get_text(api_url, timeout=timeout)
            if curl_text:
                return curl_text
        errors.append(f"europepmc_fullTextXML:{resp.status_code}")
    except Exception as e:
        errors.append(f"europepmc_fullTextXML:{repr(e)}")

    # 2) NCBI OAI-PMH fallback
    pmcid_num = re.sub(r"^PMC", "", pmcid, flags=re.I)
    oai_url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        f"?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc"
    )
    try:
        resp2 = requests.get(oai_url, timeout=timeout, headers=headers)
        if resp2.ok and resp2.text.strip():
            return resp2.text
        if resp2.status_code == 403:
            curl_text = _curl_get_text(oai_url, timeout=timeout)
            if curl_text:
                return curl_text
        errors.append(f"ncbi_oai:{resp2.status_code}")
    except Exception as e:
        errors.append(f"ncbi_oai:{repr(e)}")

    # 3) PMC Open Access API fallback: download OA package and extract .nxml
    oa_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        oa_resp = requests.get(oa_api, timeout=timeout, headers=headers)
        oa_text = oa_resp.text if (oa_resp.ok and oa_resp.text) else ""
        if (not oa_text) and oa_resp.status_code == 403:
            oa_text = _curl_get_text(oa_api, timeout=timeout) or ""
        if oa_text:
            oa_soup = BeautifulSoup(oa_text, "xml") if BeautifulSoup is not None else None
            tgz_url = None
            if oa_soup is not None:
                # Try strict format first.
                link = oa_soup.find("link", {"format": "tgz"})
                if link and link.get("href"):
                    tgz_url = link.get("href")
                # Then relaxed search by href suffix.
                if not tgz_url:
                    for lk in oa_soup.find_all("link"):
                        href = lk.get("href")
                        if href and re.search(r"\.(?:tgz|tar\.gz)$", href, flags=re.I):
                            tgz_url = href
                            break

            # Final fallback: regex over raw XML.
            if not tgz_url:
                m = re.search(r'href="([^"]+\.(?:tgz|tar\.gz))"', oa_text, flags=re.I)
                if m:
                    tgz_url = m.group(1)

            if tgz_url:
                    # OA API often returns ftp URLs; convert to https mirror.
                    if tgz_url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
                        tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", 1)
                    tgz_resp = requests.get(tgz_url, timeout=timeout, headers=headers)
                    tgz_resp.raise_for_status()
                    with tarfile.open(fileobj=io.BytesIO(tgz_resp.content), mode="r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.lower().endswith(".nxml"):
                                f = tar.extractfile(member)
                                if f is not None:
                                    xml_bytes = f.read()
                                    return xml_bytes.decode("utf-8", errors="ignore")
        errors.append(f"pmc_oa_api:{oa_resp.status_code if 'oa_resp' in locals() else 'unknown'}")
    except Exception as e:
        errors.append(f"pmc_oa_api:{repr(e)}")

    raise RuntimeError(f"Unable to fetch full text XML for {pmcid}. Fallback errors: {'; '.join(errors)}")


def parse_html_table(table_tag) -> List[List[str]]:
    """
    Parse a table with rowspan/colspan into a fully expanded 2D matrix.
    """
    rows_out: List[List[Optional[str]]] = []
    trs = table_tag.find_all("tr")

    for r_idx, tr in enumerate(trs):
        while len(rows_out) <= r_idx:
            rows_out.append([])
        row = rows_out[r_idx]

        cells = tr.find_all(["th", "td"])
        c_idx = 0
        for cell in cells:
            while c_idx < len(row) and row[c_idx] is not None:
                c_idx += 1

            txt = _clean_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for rr in range(r_idx, r_idx + rowspan):
                while len(rows_out) <= rr:
                    rows_out.append([])
                target_row = rows_out[rr]
                while len(target_row) < c_idx + colspan:
                    target_row.append(None)
                for cc in range(c_idx, c_idx + colspan):
                    if target_row[cc] is None:
                        target_row[cc] = txt

            c_idx += colspan

    max_len = max((len(r) for r in rows_out), default=0)
    final_rows: List[List[str]] = []
    for r in rows_out:
        padded = r + [None] * (max_len - len(r))
        final_rows.append([x if x is not None else "" for x in padded])
    return final_rows


def table_to_dataframe(table_tag) -> pd.DataFrame:
    matrix = parse_html_table(table_tag)
    if not matrix:
        return pd.DataFrame()
    return pd.DataFrame(matrix)


def sanitize_sheet_name(name: str, fallback: str) -> str:
    name = _clean_text(name)
    if not name:
        name = fallback
    name = re.sub(r"[\\/*?:\[\]]", "_", name)
    return name[:31]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
            flat_cols.append(" | ".join(parts) if parts else "")
        out = df.copy()
        out.columns = flat_cols
        return out
    return df


def pick_table(soup, table_id: Optional[str], table_selector: Optional[str], table_index: int):
    if table_id:
        t = soup.find("table", {"id": table_id})
        if t is None:
            wrap = soup.find("table-wrap", {"id": table_id})
            if wrap is not None:
                t = wrap.find("table")
        if t is None:
            # case-insensitive id fallback
            tid = table_id.lower()
            t = soup.find("table", id=lambda x: isinstance(x, str) and x.lower() == tid)
            if t is None:
                wrap = soup.find("table-wrap", id=lambda x: isinstance(x, str) and x.lower() == tid)
                if wrap is not None:
                    t = wrap.find("table")
        if t is None:
            raise ValueError(f"Cannot find table with id={table_id}")
        return [t]

    if table_selector:
        tables = soup.select(table_selector)
    else:
        tables = soup.find_all("table")

    if not tables:
        raise ValueError("No <table> found on page.")

    if table_index < 0 or table_index >= len(tables):
        raise ValueError(f"table_index {table_index} is out of range. Found {len(tables)} tables.")

    return [tables[table_index]]


def main():
    parser = argparse.ArgumentParser(description="Download table(s) from a URL and save as Excel.")
    parser.add_argument("--url", required=True, help="Web page URL that contains table(s)")
    parser.add_argument("--out", default="table_from_url.xlsx", help="Output xlsx path")
    parser.add_argument("--table_id", default=None, help="Specific table id attribute to target")
    parser.add_argument("--table_selector", default=None, help="CSS selector for table(s), e.g. '#Tab2 table'")
    parser.add_argument("--table_index", type=int, default=0, help="0-based table index if multiple tables exist")
    parser.add_argument("--all_tables", action="store_true", help="Export all matched tables into separate sheets")
    args = parser.parse_args()

    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed. Run: python3 -m pip install beautifulsoup4")

    pmcid, url_table_id = _extract_pmc_info(args.url)
    resolved_table_id = args.table_id or url_table_id

    # Fast-path fallback for PMC direct table assets (often bypasses page-level 403).
    if pmcid and resolved_table_id:
        direct_tables = _try_pmc_direct_table_download(pmcid, resolved_table_id)
        if direct_tables:
            with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
                for i, df in enumerate(direct_tables, start=1):
                    df2 = _flatten_columns(df)
                    df2.to_excel(writer, index=False, sheet_name=sanitize_sheet_name("", f"table_{i}"))
            print(f"Saved {len(direct_tables)} table(s) to {args.out}")
            return

    html = ""
    soup = None
    parse_with_xml = False
    try:
        html = fetch_html(args.url)
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        # PMC pages may return 403 to scripts; fallback to Europe PMC XML.
        if pmcid:
            xml = fetch_pmc_fulltext_xml(pmcid)
            soup = BeautifulSoup(xml, "xml")
            parse_with_xml = True
        else:
            raise e

    if parse_with_xml and not resolved_table_id and url_table_id:
        resolved_table_id = url_table_id

    if resolved_table_id:
        tables = pick_table(soup, table_id=resolved_table_id, table_selector=None, table_index=args.table_index)
    elif args.all_tables:
        if parse_with_xml:
            wraps = soup.find_all("table-wrap")
            tables = [w.find("table") for w in wraps if w.find("table") is not None]
        else:
            tables = soup.select(args.table_selector) if args.table_selector else soup.find_all("table")
        if not tables:
            raise ValueError("No matched tables found.")
    else:
        tables = pick_table(
            soup,
            table_id=None,
            table_selector=args.table_selector,
            table_index=args.table_index,
        )

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for i, table in enumerate(tables, start=1):
            df = table_to_dataframe(table)
            caption = ""
            cap = table.find("caption")
            if cap:
                caption = _clean_text(cap.get_text(" ", strip=True))
            sheet = sanitize_sheet_name(caption, fallback=f"table_{i}")
            df2 = _flatten_columns(df)
            df2.to_excel(writer, index=False, header=False, sheet_name=sheet)

    print(f"Saved {len(tables)} table(s) to {args.out}")


if __name__ == "__main__":
    main()
