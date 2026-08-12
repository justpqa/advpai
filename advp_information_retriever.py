from typing import Optional, List, Tuple, Dict
import re
import ast
import os
from copy import deepcopy
from itertools import chain
from lxml import etree
from xml.etree import ElementTree as ET
import requests
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, AutoModel
from llama_cpp import Llama, LogitsProcessorList
from huggingface_hub import login
from utils import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
load_dotenv()

LLAMA_CLIENT = OpenAI(base_url=f"http://localhost:{os.environ.get('LLAMA_URL_PORT', 0)}/v1", api_key="none")

ENTREZ_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

class MedCPTArticleEmbeddings:
    """Chroma-compatible embedding function using MedCPT Article Encoder."""
    def __init__(self, model_name="ncbi/MedCPT-Article-Encoder", device="mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Article encoder expects [title, abstract] pairs.
        # We have chunks with no separate title, so pass "" as title.
        pairs = [["", t] for t in texts]
        with torch.no_grad():
            enc = self.tokenizer(pairs, truncation=True, padding=True,
                                 return_tensors="pt", max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            embeds = self.model(**enc).last_hidden_state[:, 0, :]
        return embeds.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        # This should not be called for article encoder, but Chroma requires it.
        return self.embed_documents([text])[0]

class MedCPTQueryEmbeddings:
    """Chroma-compatible embedding function using MedCPT Query Encoder."""
    def __init__(self, model_name="ncbi/MedCPT-Query-Encoder", device="mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            enc = self.tokenizer(texts, truncation=True, padding=True,
                                 return_tensors="pt", max_length=64)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            embeds = self.model(**enc).last_hidden_state[:, 0, :]
        return embeds.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class AllowedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed_token_ids)] = 0
        return scores + mask

class AllowedTokensProcessorLlamaCpp:
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        import numpy as np
        mask = np.full_like(scores, float("-inf"))
        for token_id in self.allowed_token_ids:
            mask[token_id] = 0
        return scores + mask

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove tabs and newlines
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Normalise unicode minus/dash variants to ASCII hyphen before stripping
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    # Remove special characters; keep punctuation needed for numeric/scientific values
    text = re.sub(r'[^a-z0-9\s\.\,\-\+\%\(\)\<\>\=\:]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

def ingest_doc_from_pmc(pmid: int, pmcid: str, embedding_model_name: str = "NeuML/pubmedbert-base-embeddings", 
                        chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "advp2", 
                        chunk_size: int = 500, chunk_overlap: int = 50):
    documents = []
    metadata = []
    try:
        curr_doc = ""
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for d in data:
                doc = d["documents"]
                for p in doc:
                    passage = p["passages"]
                    for item in passage:
                        if "text" in item:
                            curr_doc += "\n\n" + item["text"]
        documents.append(curr_doc)
        metadata.append({"PMID": str(pmid), "PMCID": pmcid})
    except:
        return 0
        # raise Exception(f"Failed to extract paper {pmid}_{pmcid} from PMC with error {e}")
    

    # split documents into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
        splitted_documents = [clean_document(doc) for doc in splitted_documents]
    except:
        return 0
    

    # create Chroma vector store
    try:
        chroma_db = Chroma(
            persist_directory=chroma_db_path,
            # embedding_function=MedCPTArticleEmbeddings(),
            embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
            collection_name=chroma_db_collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # delete current collection contents before adding new documents
        collection = chroma_db._collection
        all_docs = collection.get(include=[])
        all_ids = all_docs["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        # add documents to Chroma vector store
        chroma_db.add_documents(splitted_documents)
        return len(splitted_documents)
    except:
        return 0
    
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
 
 
def fetch_pmc_xml(pmcid: str) -> bytes:
    """Fetch full-text JATS XML for a PMC article."""
    pmcid = pmcid.replace("PMC", "")
    url = f"{EUTILS}/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "rettype": "xml"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.content
 
 
def get_text_with_offsets(elem) -> tuple:
    """
    Flatten an element to plain text while tracking, for each <xref ref-type='bibr'>,
    (rid, start_char, end_char, marker_text) in the flat string.
    """
    parts = []
    xrefs = []  # (rid, start, end, marker)
    pos = [0]
 
    def walk(node):
        if node.text:
            parts.append(node.text)
            pos[0] += len(node.text)
        for child in node:
            if child.tag == "xref" and child.get("ref-type") == "bibr":
                start = pos[0]
                marker = "".join(child.itertext())
                parts.append(marker)
                pos[0] += len(marker)
                end = pos[0]
                rid = child.get("rid", "")
                # rid can be space-separated for grouped citations
                for r in rid.split():
                    xrefs.append((r, start, end, marker))
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)
                pos[0] += len(child.tail)
 
    walk(elem)
    return "".join(parts), xrefs
 
 
def extract_context(
    text: str,
    start: int,
    end: int,
    mode: str = "sentence",
    window: int = 500,
) -> str:
    """
    Extract context around a citation at [start, end) in text.
 
    mode options:
      "chars"    — raw character window of size `window` on each side; no snapping
      "sentence" — (default) expand to nearest sentence boundaries within `window` chars
      "sentences"— N full sentences before+after; pass window=N (default 1)
      "paragraph"— the entire containing paragraph (ignores `window`)
    """
    if mode == "chars":
        left = max(0, start - window)
        right = min(len(text), end + window)
 
    elif mode == "sentence":
        left = max(0, start - window)
        right = min(len(text), end + window)
        m = re.search(r"[.!?]\s+[A-Z]", text[left:start])
        if m:
            left = left + m.end() - 1
        m = re.search(r"[.!?](?:\s|$)", text[end:right])
        if m:
            right = end + m.end()
 
    elif mode == "sentences":
        n = max(1, window)
        # Split on sentence-ending punctuation followed by space+capital
        sentence_ends = [0] + [
            m.end() for m in re.finditer(r"[.!?]\s+(?=[A-Z])", text)
        ] + [len(text)]
        # Find which sentence the citation lives in
        cite_sentence = next(
            (i for i, e in enumerate(sentence_ends) if e > start), 1
        ) - 1
        left_idx = max(0, cite_sentence - n)
        right_idx = min(len(sentence_ends) - 1, cite_sentence + n + 1)
        left = sentence_ends[left_idx]
        right = sentence_ends[right_idx]
 
    elif mode == "paragraph":
        left_m = text.rfind("\n\n", 0, start)
        left = left_m + 2 if left_m != -1 else 0
        right_m = text.find("\n\n", end)
        right = right_m if right_m != -1 else len(text)
 
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use: chars | sentence | sentences | paragraph")
 
    return text[left:right].strip().replace("\n", " ")
 
 
def parse_reference(ref_elem) -> Dict:
    """Pull title, PMID, PMCID, DOI, authors from a <ref> element."""
    out = {"pmid": None, "pmcid": None, "doi": None, "title": None, "authors": None}
 
    for pi in ref_elem.findall(".//pub-id"):
        t = pi.get("pub-id-type")
        if t == "pmid":
            out["pmid"] = (pi.text or "").strip()
        elif t == "pmcid" or t == "pmc":
            out["pmcid"] = (pi.text or "").strip()
        elif t == "doi":
            out["doi"] = (pi.text or "").strip()
 
    title_el = ref_elem.find(".//article-title")
    if title_el is not None:
        out["title"] = " ".join(title_el.itertext()).strip()
 
    authors = []
    for name in ref_elem.findall(".//name"):
        surname = name.findtext("surname", "").strip()
        given = name.findtext("given-names", "").strip()
        if surname:
            authors.append(f"{surname} {given}".strip())
    if authors:
        out["authors"] = authors
 
    return out
 
 
def resolve_missing_ids(refs: Dict[str, Dict]) -> None:
    """For refs missing PMID/PMCID, try NCBI's ID converter using DOI."""
    needs = [(rid, r) for rid, r in refs.items()
             if r.get("doi") and not (r.get("pmid") and r.get("pmcid"))]
    for rid, r in needs:
        try:
            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {"ids": r["doi"], "format": "json", "tool": "pmc_cite", "email": "you@example.com"}
            resp = requests.get(url, params=params, timeout=15).json()
            rec = resp.get("records", [{}])[0]
            r["pmid"] = r.get("pmid") or rec.get("pmid")
            r["pmcid"] = r.get("pmcid") or rec.get("pmcid")
            # time.sleep(0.34)  # NCBI rate limit: 3 req/s without API key
        except Exception:
            pass
 
 
def get_citations_with_context(
    pmcid: str,
    mode: str = "sentences",
    window: int = 500,
) -> List[Dict]:
    xml_bytes = fetch_pmc_xml(pmcid)
    root = etree.fromstring(xml_bytes)
 
    # Build reference dictionary keyed by ref id
    refs: Dict[str, Dict] = {}
    for ref in root.findall(".//ref-list//ref"):
        rid = ref.get("id")
        if rid:
            refs[rid] = parse_reference(ref)
 
    resolve_missing_ids(refs)
 
    # Walk body paragraphs, capture each xref + context
    results = []
    body = root.find(".//body")
    if body is None:
        return results
 
    for p in body.iter("p"):
        flat, xrefs = get_text_with_offsets(p)
        if not xrefs:
            continue
        # Section name if available
        section = None
        parent = p.getparent()
        while parent is not None:
            title = parent.find("title")
            if title is not None and title.text:
                section = title.text.strip()
                break
            parent = parent.getparent()
 
        for rid, start, end, marker in xrefs:
            ref = refs.get(rid, {})
            if ref.get("pmid", None):
                results.append({
                    "pmid": int(ref.get("pmid")),
                    "pmcid": ref.get("pmcid"),
                    "title": ref.get("title"),
                    "doi": ref.get("doi"),
                    "citation_marker": marker,
                    "section": section,
                    "context": extract_context(flat, start, end, mode=mode, window=window),
                })
    
    # remove duplicate paper
    results_no_duplicate = {}
    for res in results:
        if res["pmid"] not in results_no_duplicate:
            results_no_duplicate[res["pmid"]] = [res["pmcid"], res["title"], "", [res["context"]]]
        else:
            results_no_duplicate[res["pmid"]][-1].append(res["context"])
    
    # also extract abstract
    results_all_pmids = list(results_no_duplicate.keys())
    for i in range(0, len(results_all_pmids), 20):
        batch_pmids = results_all_pmids[i: min(i + 20, len(results_all_pmids))]
        batch_pmids = [str(pmid) for pmid in batch_pmids]
        r = requests.get(
            f"{ENTREZ_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "rettype": "xml",
                "retmode": "xml",
                "api_key": os.environ.get("ENTREZ_API_KEY", ""),
            },
            timeout=60,
        )
        r.raise_for_status()

        for article in ET.fromstring(r.content).findall(".//PubmedArticle"):
            # get article
            art = article.find(".//Article")
            # get abstract and search for term that is in there
            abstract = " ".join("".join(el.itertext()) for el in art.findall(".//AbstractText"))
            try:
                pmid = int(article.findtext(".//MedlineCitation/PMID", ""))
                if pmid in results_all_pmids:
                    results_no_duplicate[pmid][2] = abstract
            except:
                continue

    # # back to normal format
    results = []
    for pmid in results_no_duplicate:
        results.append({
            "pmid": pmid,
            "pmcid": results_no_duplicate[pmid][0],
            "title": results_no_duplicate[pmid][1],
            "abstract": results_no_duplicate[pmid][2],
            "context_lst": results_no_duplicate[pmid][3]
        })
    # # extract 
    return results


GWAS_TIAB_TERMS = [
    'GWAS', 'genome-wide association', "polygenic risk", "polygenic score", "genetic variant",
    "genetic risk", "SNP", "single nucleotide polymorphism", "risk loci", "genetic association",
    "meta-analysis", "Mendelian randomization", "whole genome sequencing", "whole exome sequencing",
    "copy number variant", "heritability", "gene expression", "transcriptome", "eQTL",
    "epigenome", "DNA methylation", "Genetic Predisposition to Disease", "Genetic Loci", 
    "Aged", "Humans", "Risk Factors", "genome-wide", "apoe", "late-onset", "polymorphisms", "genotype",
    "pathogenesis", "pathology", "biomarkers", "haplotype", "allete", "chromosome", "methylation", "ε4", 
    "phenotype", "linkage", "neuroimaging", "polygenic", "quantitative trait", "epigenetic"
]
AD_TIAB_TERMS = [
    "Alzheimer", "dementia", "cognitive impairment", "neurodegeneration", "Frontotemporal dementia", 
    "Parkinson Disease", "Amyotrophic Lateral Sclerosis", "Progressive Supranuclear Palsy", 
    "Corticobasal Degeneration", "Vascular dementia", "AD", "neurodegenerative", "parkinson", "cognitive",
    "amyloid", "sclerosis", "cognitive decline", "cerebrospinal", "hippocampal"
]

def get_gwas_ad_citations_with_context(
    pmcid: str,
    mode: str = "sentences",
    window: int = 500,
) -> List[Dict]:
    res = get_citations_with_context(pmcid, mode, window)
    gwas_ad_res = []
    for item in res:
        is_gwas = (
            any((term.lower() in str(item["title"]).lower() for term in GWAS_TIAB_TERMS)) or
            any((term.lower() in str(item["abstract"]).lower() for term in GWAS_TIAB_TERMS))
        )
        is_ad = (
            any((term.lower() in str(item["title"]).lower() for term in AD_TIAB_TERMS)) or
            any((term.lower() in str(item["abstract"]).lower() for term in AD_TIAB_TERMS))
        )
        if is_gwas and is_ad:
            gwas_ad_res.append(item)
    return gwas_ad_res

class ADVPInformationRetriever:
    def __init__(self, referencing_col_df: pd.DataFrame,
                 chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "advp2", 
                 embeddings_model_name: str = "NeuML/pubmedbert-base-embeddings", reranker_model_name: str = "BAAI/bge-reranker-base",
                 llm_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct", llm_gguf_path: Optional[str] = "./model/qwen2.5-3b-instruct-q8/qwen2.5-3b-instruct-q8_0.gguf",
                 use_hf: bool = True, device: Optional[str] = None):
        # load ref col df
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        # Definition-only context (no examples) — safe to show to the LLM.
        self.referencing_col_context_lst = referencing_col_df.apply(
            lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"],
            axis=1,
        ).to_list()
        # Examples kept separate; used ONLY to strengthen retrieval and as a
        # labeled, anti-leakage hint block inside the prompt.
        if "examples" in referencing_col_df.columns:
            self.referencing_col_examples_lst = referencing_col_df["examples"].apply(
                lambda x: x.strip() if isinstance(x, str) and x.strip() else ""
            ).to_list()
        else:
            self.referencing_col_examples_lst = ["" for _ in self.referencing_col_lst]
        self.referencing_col_use_examples_in_llm_lst = referencing_col_df["use_examples_in_llm"]
        # Retrieval query = definition + examples (examples help embedding recall,
        # but will NOT be shown verbatim to the LLM in the generation prompt).
        self.referencing_col_retrieval_query_lst = [
            ctx if not ex else f"{ctx} Examples: {ex}."
            for ctx, ex in zip(self.referencing_col_context_lst, self.referencing_col_examples_lst)
        ]

        # load vector store
        self.vector_store = Chroma(
            persist_directory=chroma_db_path,
            # embedding_function=MedCPTQueryEmbeddings(),
            embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_name),
            collection_name=chroma_db_collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
        # load the device
        self.device = device if device is not None else "cpu"
        
        # load the embeddings
        self.embeddings_model_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
        self.embeddings_model = AutoModel.from_pretrained(
            embeddings_model_name
        )

        # load the reranker
        # self.reranker_model = CrossEncoder(
        #     model_name_or_path=reranker_model_name, 
        #     device = self.device,
        #     trust_remote_code = True
        # ) 
        self.reranker_model_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.reranker_model.eval()

        # load the llm
        # bnb_config = BitsAndBytesConfig(
        #     # load_in_8bit=True
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16, 
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        if use_hf and llm_model_name is not None:
            self.use_hf = True
            login(os.environ.get("HF_TOKEN", ""))
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                # quantization_config = bnb_config,
                device_map = self.device,
                # torch_dtype=torch.bfloat16,
            )
            self.model.eval()
            allowed_chars = list("01")
            allowed_token_ids = set()
            for token, token_id in self.tokenizer.get_vocab().items():
                if all(c in allowed_chars for c in token):
                    allowed_token_ids.add(token_id)
            self.allowed_tokens_processor = AllowedTokensProcessor(allowed_token_ids)
        elif not use_hf and llm_gguf_path is not None:
            self.use_hf = False
            self.llm = Llama(
                model_path=llm_gguf_path,
                n_ctx=16384,
                n_gpu_layers=-1,
                verbose=False,
            )
            allowed_token_ids = set()
            for token_id in range(self.llm.n_vocab()):
                token_bytes = self.llm.detokenize([token_id])
                token_str = token_bytes.decode("utf-8", errors="replace")
                if all(c in "01" for c in token_str) and len(token_str) > 0:
                    allowed_token_ids.add(token_id)
            self.allowed_tokens_processor = AllowedTokensProcessorLlamaCpp(allowed_token_ids)
        else:
            raise Exception("Missing either llm_model_name (if use_hf=True) or llm_gguf_path (if use_hf=False)")
        # NOTE: temporary use only hf model since we try logit bias to only generate certain ans
        # self.use_hf = True
        # login(os.environ.get("HF_TOKEN", ""))
        # self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     llm_model_name,
        #     # quantization_config = bnb_config,
        #     device_map = "auto",
        #     # torch_dtype=torch.bfloat16,
        # )
        # # self.model.to(self.device)
        # self.model.eval()

        # logit processor for choices
        # allowed_chars = list("01")
        # allowed_token_ids = set()
        # for token, token_id in self.tokenizer.get_vocab().items():
        #     if all(c in allowed_chars for c in token):
        #         allowed_token_ids.add(token_id)
        # self.allowed_tokens_processor = AllowedTokensProcessor(allowed_token_ids)


        # NOTE: config for search and generate, add it as params later
        self.top_k = 20
        self.top_k_rerank = 5
        self.max_new_tokens = 32
        self.similarity_score_threshold = 0.0
        self.temperature = 0
        self.top_p = 1
    
    # def make_embeddings(self, sentences: str | List[str]) -> torch.Tensor:
    #     inputs = self.embeddings_model_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     # get token embeddings
    #     with torch.no_grad():
    #         output = self.embeddings_model(**inputs)
    #     token_embeddings = output[0]

    #     # extract mask and mean pooling for sentence embeddings
    #     input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    #     embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    #     # final normalization
    #     embeddings = F.normalize(embeddings, p=2, dim=1)

    #     # size (# senetences, # dim)
    #     return embeddings

    # def calculate_best_choices(self, info_lst: str | List[str], choices_lst: str | List[str], threshold: float = 0.4) -> List[str]:
    #     embeddings_info, embeddings_choices = self.make_embeddings(info_lst), self.make_embeddings(choices_lst)
    #     similarity_score = embeddings_info @ embeddings_choices.T
    #     print(info_lst)
    #     print(choices_lst)
    #     print(similarity_score)
    #     if similarity_score.max() < threshold:
    #         return []
    #     best_score_by_choice = torch.max(similarity_score, dim=0).values
    #     print(best_score_by_choice)
    #     return [c for i, c in enumerate(choices_lst) if best_score_by_choice[i] >= threshold]

    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        pairs = [[query, chunk] for chunk in chunks]
        inputs = self.reranker_model_tokenizer(
            pairs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze(-1)
        
        ranked = sorted(
            zip(chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        
        ranked_docs = [chunk for chunk, _ in ranked]
        return ranked_docs[:self.top_k_rerank] if self.top_k_rerank else ranked_docs
    
    def make_messages(self, query: str, documents: List[str], examples: str = "", use_examples_in_llm: bool = True) -> List[Dict]:
        document_str = "\n\n".join([f"EXCERPT {i + 1}:\n{d}" for i, d in enumerate(documents)])

        # Examples from the CSV are quarantined in a clearly labeled block and
        # explicitly forbidden unless they also appear in the EXCERPTs. This
        # keeps them available as weak context without encouraging the model
        # to regurgitate them as extractions.
        hints_block = (
            f"\nRetrieval hints (DO NOT output any of these unless they appear verbatim in the EXCERPTs above):\n{examples}\n"
            if examples and use_examples_in_llm else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict biomedical information extraction engine. "
                    "Only output values that appear verbatim in the provided EXCERPTs. "
                    "Do not copy any term from the field definition, from retrieval hints, "
                    "or from your own domain knowledge if it is not literally present in the EXCERPTs. "
                    "If the EXCERPTs do not support a value, return an empty list. "
                    "Respond with a single JSON object and nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"""Goal: extract candidate values for a field, grounded strictly in the EXCERPTs.

Field:
{query}
{hints_block}
EXCERPTs:
{document_str}

Rules:
- Only include items that appear literally in the EXCERPTs (exact casing, exact spelling).
- No paraphrasing, no expansions, no translations.
- If a long name and an abbreviation both appear in the EXCERPTs, include BOTH.
- De-duplicate items.
- If nothing in the EXCERPTs supports the field, return {{"items": []}}.

Respond with a single JSON object only, no prose, no markdown fence:
{{"items": ["<verbatim string from EXCERPT>", ...]}}"""
            },
        ]
        return messages

    def make_prompt(self, query: str, documents: List[str]) -> str:    
        messages = self.make_messages(query, documents)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    

    def make_messages_with_choice(self, query: str, documents: List[str], choice: str) -> List[Dict]:
        document_str = "\n\n".join([f"EXCERPT {i + 1}:\n{d}" for i, d in enumerate(documents)])

        messages = [
            {
                "role": "system",
                "content": "You are a strict biomedical evidence verifier. Only use the provided excerpts. Never guess. Output only 0 or 1."
            },
            {
                "role": "user",
                "content": f"""Goal: decide whether the evidence excerpts clearly and explicitly support the candidate answer for the given field.

Field:
{query}

Candidate answer:
{choice}

Evidence text:
{document_str}

Rules:
- Output 1 if the excerpts clearly and explicitly support the candidate answer.
- Output 0 if the excerpts do not support it, or if the evidence is ambiguous.
- Output only a single digit: 0 or 1. Nothing else.

Example:
Field: Type of association analysis
Candidate answer: gene-based
Evidence: We performed a gene-based test aggregating rare variants per gene.
Output: 1

Output: """
            }
        ]
        return messages

    def make_prompt_lst_with_choices(self, query: str, documents: List[str], choices: List[str]) -> List[str]:
        prompt_lst = []
        for choice in choices:
            messages = self.make_messages_with_choice(query, documents, choice)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_lst.append(prompt)
        return prompt_lst
    
    def extract_lst_from_llm_output(self, text: str) -> List[str]:
        text = text.replace("```json", "").replace("```", "").strip()
        # Prefer the structured JSON object produced by the new prompt.
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            import json
            try:
                obj = json.loads(json_match.group(0))
                items = obj.get("items", []) if isinstance(obj, dict) else []
                if isinstance(items, list):
                    lst = [str(x) for x in items if isinstance(x, (str, int, float))]
                    lst = list(set([item.lower() for item in lst]))
                    return lst
            except Exception:
                pass
        # Backward-compatible fallback for bare Python list outputs.
        matches = re.findall(r"\[.*?\]", text, re.DOTALL)
        if not matches:
            return []
        try:
            lst = ast.literal_eval(matches[-1])
            lst = list(set([item.lower() for item in lst]))
            return lst
        except Exception:
            return []
        
    def extract_lst_from_llm_output_choices(self, text: str) -> List[int]:
        # Remove code fences and trim
        text = text.replace("```", "").strip()

        # Extract the last bracketed list
        matches = re.findall(r"\[[^\]]*\]", text)
        if not matches:
            return []

        list_str = matches[-1]

        # Extract all integers inside the brackets
        numbers = re.findall(r"\d+", list_str)

        return [int(n) for n in numbers]
    
    def extract_possible_info_from_paper_specified(
        self, pmid: int, pmcid: str, mode: str = "chunk",
        ref_col_lst: List = [], ref_col_context_lst: List = [],
        ref_col_examples_lst: List = [], ref_col_use_examples_in_llm_lst: List = [],
        ref_col_retrieval_query_lst: List = []
    ):
        """
        Given a paper, extract all possible answer for each category, this is the version that require you to specify the list
        """
        res = {}

        num_docs = ingest_doc_from_pmc(pmid, pmcid)
        if num_docs == 0:
            return {ref_col: [] for ref_col in ref_col_lst}
        
        for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query in zip(
            ref_col_lst,
            ref_col_context_lst,
            ref_col_examples_lst,
            ref_col_use_examples_in_llm_lst,
            ref_col_retrieval_query_lst,
        ):
            # Retrieval uses definition + examples (better recall); the LLM
            # prompt sees only the definition, with examples in a quarantined block.
            query = ref_col_context
            retrieval_query = ref_col_retrieval_query
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = retrieval_query,
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            # scores = self.reranker_model.predict([(retrieval_query, d) for d in documents])
            # documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            # documents = documents[:self.top_k_rerank]
            documents = self.rerank(retrieval_query, documents)
            messages = self.make_messages(query, documents, examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
            response = LLAMA_CLIENT.chat.completions.create(
                model="local",
                messages=messages, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["items"],
                    },
                },
            )
            content = response.choices[0].message.content
            if ref_col not in res:
                res[ref_col] = []
            new_info = self.extract_lst_from_llm_output(content)
            new_info = list(map(lambda x: x.lower(), new_info))
            res[ref_col] = list(set(res[ref_col] + new_info))
            if mode == "all":
                messages = self.make_messages(query, [documents], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                response = LLAMA_CLIENT.chat.completions.create(
                    model="local",
                    messages=messages, max_tokens=self.max_new_tokens,
                    temperature=self.temperature, top_p=self.top_p,
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["items"],
                        },
                    },
                )
                content = response.choices[0].message.content
                # content = response["choices"][0]["message"]["content"]
                if ref_col not in res:
                    res[ref_col] = []
                new_info = self.extract_lst_from_llm_output(content)
                new_info = list(map(lambda x: x.lower(), new_info))
                res[ref_col] = list(set(res[ref_col] + new_info))
            elif mode == "chunk":
                for doc in documents:
                    messages = self.make_messages(query, [doc], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                    response = LLAMA_CLIENT.chat.completions.create(
                        model="local",
                        messages=messages, max_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        response_format={
                            "type": "json_object",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["items"],
                            },
                        },
                    )
                    content = response.choices[0].message.content
                    # content = response["choices"][0]["message"]["content"]
                    if ref_col not in res:
                        res[ref_col] = []
                    new_info = self.extract_lst_from_llm_output(content)
                    new_info = list(map(lambda x: x.lower(), new_info))
                    res[ref_col] = list(set(res[ref_col] + new_info))

        return res

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str, mode: str = "chunk") -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        num_docs = ingest_doc_from_pmc(pmid, pmcid)
        if num_docs == 0:
            return {ref_col: [] for ref_col in self.referencing_col_lst}

        for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query in zip(
            self.referencing_col_lst,
            self.referencing_col_context_lst,
            self.referencing_col_examples_lst,
            self.referencing_col_use_examples_in_llm_lst,
            self.referencing_col_retrieval_query_lst,
        ):
            # Retrieval uses definition + examples (better recall); the LLM
            # prompt sees only the definition, with examples in a quarantined block.
            query = ref_col_context
            retrieval_query = ref_col_retrieval_query
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = retrieval_query,
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            # scores = self.reranker_model.predict([(retrieval_query, d) for d in documents])
            # documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            # documents = documents[:self.top_k_rerank]
            documents = self.rerank(retrieval_query, documents)
            messages = self.make_messages(query, documents, examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
            response = LLAMA_CLIENT.chat.completions.create(
                model="local",
                messages=messages, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["items"],
                    },
                },
            )
            content = response.choices[0].message.content
            if ref_col not in res:
                res[ref_col] = []
            new_info = self.extract_lst_from_llm_output(content)
            new_info = list(map(lambda x: x.lower(), new_info))
            res[ref_col] = list(set(res[ref_col] + new_info))
            if mode == "all":
                messages = self.make_messages(query, [documents], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                # response = self.llm.create_chat_completion(
                #     messages=messages,
                #     max_tokens=self.max_new_tokens,
                #     temperature=self.temperature,
                #     top_p=self.top_p,
                #     response_format={
                #         "type": "json_object",
                #         "schema": {
                #             "type": "object",
                #             "properties": {
                #                 "items": {
                #                     "type": "array",
                #                     "items": {"type": "string"},
                #                 },
                #             },
                #             "required": ["items"],
                #         },
                #     },
                # )
                response = LLAMA_CLIENT.chat.completions.create(
                    model="local",
                    messages=messages, max_tokens=self.max_new_tokens,
                    temperature=self.temperature, top_p=self.top_p,
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["items"],
                        },
                    },
                )
                content = response.choices[0].message.content
                # content = response["choices"][0]["message"]["content"]
                if ref_col not in res:
                    res[ref_col] = []
                new_info = self.extract_lst_from_llm_output(content)
                new_info = list(map(lambda x: x.lower(), new_info))
                res[ref_col] = list(set(res[ref_col] + new_info))
            elif mode == "chunk":
                for doc in documents:
                    messages = self.make_messages(query, [doc], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                    # response = self.llm.create_chat_completion(
                    #     messages=messages,
                    #     max_tokens=self.max_new_tokens,
                    #     temperature=self.temperature,
                    #     top_p=self.top_p,
                    #     response_format={
                    #         "type": "json_object",
                    #         "schema": {
                    #             "type": "object",
                    #             "properties": {
                    #                 "items": {
                    #                     "type": "array",
                    #                     "items": {"type": "string"},
                    #                 },
                    #             },
                    #             "required": ["items"],
                    #         },
                    #     },
                    # )
                    response = LLAMA_CLIENT.chat.completions.create(
                        model="local",
                        messages=messages, max_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        response_format={
                            "type": "json_object",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["items"],
                            },
                        },
                    )
                    content = response.choices[0].message.content
                    # content = response["choices"][0]["message"]["content"]
                    if ref_col not in res:
                        res[ref_col] = []
                    new_info = self.extract_lst_from_llm_output(content)
                    new_info = list(map(lambda x: x.lower(), new_info))
                    res[ref_col] = list(set(res[ref_col] + new_info))


            # all_messages = []
            # for doc in documents:
            #     messages = self.make_messages(
            #         query, [doc],
            #         examples=ref_col_examples,
            #         use_examples_in_llm=ref_col_use_examples_in_llm
            #     )
            #     all_messages.append(messages)

            # Single LLM call
            # async def call_llm(doc):
            #     messages = self.make_messages(
            #         query, [doc],
            #         examples=ref_col_examples,
            #         use_examples_in_llm=ref_col_use_examples_in_llm
            #     )
            #     response = await ASYNC_LLAMA_CLIENT.chat.completions.create(
            #         model="local",
            #         messages=messages,
            #         max_tokens=self.max_new_tokens,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #         response_format={"type": "json_object"},
            #     )
            #     return response.choices[0].message.content

            # # Run in parallel, chunk_size should match --parallel on server
            # async def call_all(documents, chunk_size=4):
            #     results = []
            #     for i in range(0, len(documents), chunk_size):
            #         chunk = documents[i:min(i + chunk_size, len(documents))]
            #         tasks = [call_llm(m) for m in chunk]
            #         chunk_results = await asyncio.gather(*tasks)
            #         results.extend(chunk_results)
            #     return results

            # # Blocks until all done
            # contents = asyncio.run(call_all(documents, chunk_size=4))

            # # contents = []
            # # for i in range(0, len(all_messages), 2):
            # #     batch_messages = all_messages[i: min(i + 2, len(all_messages))]
            # #     batch_contents = asyncio.run(call_all(batch_messages))
            # #     contents.extend(batch_contents)
            # for content in contents:
            #     if ref_col not in res:
            #         res[ref_col] = []
            #     new_info = self.extract_lst_from_llm_output(content)
            #     new_info = list(map(lambda x: x.lower(), new_info))
            #     res[ref_col] = list(set(res[ref_col] + new_info))

            # for doc in documents:
            #     messages = self.make_messages(
            #         query, [doc],
            #         examples=ref_col_examples,
            #         use_examples_in_llm=ref_col_use_examples_in_llm,
            #     )
            #     response = LLAMA_CLIENT.chat.completions.create(
            #         model="local",
            #         messages=messages,
            #         max_tokens=self.max_new_tokens,
            #         temperature=self.temperature,
            #         top_p=self.top_p,
            #         response_format={"type": "json_object"},
            #     )
            #     content = response.choices[0].message.content

            #     if ref_col not in res:
            #         res[ref_col] = []
            #     new_info = self.extract_lst_from_llm_output(content)
            #     new_info = list(map(lambda x: x.lower(), new_info))
            #     res[ref_col] = list(set(res[ref_col] + new_info))

        return res
    
    def extract_possible_info_from_paper_and_citations(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = self.extract_possible_info_from_paper(pmid, pmcid, mode = "chunk")   

        # now search from citations — only keep those with a resolvable pmcid
        possible_citations = get_gwas_ad_citations_with_context(pmcid)
        possible_citations = [c for c in possible_citations if c.get("pmcid")]
        if len(possible_citations) > 0:
            pmid_pmcid_to_params_lst = {}
            threshold = 0.5
            for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query in zip(
                self.referencing_col_lst,
                self.referencing_col_context_lst,
                self.referencing_col_examples_lst,
                self.referencing_col_use_examples_in_llm_lst,
                self.referencing_col_retrieval_query_lst,
            ):
                for citation in possible_citations:
                    context_lst = citation["context_lst"]
                    context_query_similarity = calculate_similarity_scores(context_lst, [ref_col_retrieval_query], self.embeddings_model, self.embeddings_model_tokenizer)
                    if torch.max(context_query_similarity).item() >= threshold:
                        pmid, pmcid = citation["pmid"], citation["pmcid"]
                        if (pmid, pmcid) not in pmid_pmcid_to_params_lst:
                            pmid_pmcid_to_params_lst[(pmid, pmcid)] = {
                                "ref_col_lst": [], "ref_col_context_lst": [], "ref_col_examples_lst": [],
                                "ref_col_use_examples_in_llm_lst": [], "ref_col_retrieval_query_lst": []
                            }
                        pmid_pmcid_to_params_lst[(pmid, pmcid)]["ref_col_lst"].append(ref_col)
                        pmid_pmcid_to_params_lst[(pmid, pmcid)]["ref_col_context_lst"].append(ref_col_context)
                        pmid_pmcid_to_params_lst[(pmid, pmcid)]["ref_col_examples_lst"].append(ref_col_examples)
                        pmid_pmcid_to_params_lst[(pmid, pmcid)]["ref_col_use_examples_in_llm_lst"].append(ref_col_use_examples_in_llm)
                        pmid_pmcid_to_params_lst[(pmid, pmcid)]["ref_col_retrieval_query_lst"].append(ref_col_retrieval_query)
            
            print(f"Number of extra papers used: {len(pmid_pmcid_to_params_lst)}")
            for pmid, pmcid in pmid_pmcid_to_params_lst:
                try:
                    new_res = self.extract_possible_info_from_paper_specified(
                        pmid = pmid, pmcid = pmcid, mode = "chunk",
                        **pmid_pmcid_to_params_lst[(pmid, pmcid)]
                    )
                    for ref_col in new_res:
                        res[ref_col] = list(set(res[ref_col] + new_res[ref_col]))
                except Exception as e:
                    continue

        return res
    
    def extract_possible_info_from_paper_with_choices(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category.
        - If no choices: falls back to the same free-form extraction as extract_possible_info_from_paper.
        - If choices exist: for each choice, asks the LLM to return 1 (valid) or 0 (invalid) using the
          logits processor, then collects all choices the LLM marked as valid.
        """
        res = {}

        for ref_col, ref_col_context, ref_col_choices in zip(self.referencing_col_lst, self.referencing_col_context_lst, self.referencing_col_choices_lst):
            query = ref_col_context
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=self.top_k,
                filter={"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = self.reranker_model.predict([(query, d) for d in documents])
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            if len(ref_col_choices) > 0:
                # For each choice, verify if it is supported by the documents (1 = valid, 0 = invalid)
                if self.use_hf:
                    prompt_lst = self.make_prompt_lst_with_choices(query, documents, ref_col_choices)
                    inputs = self.tokenizer(prompt_lst, padding=True, padding_side="left", truncation=True, return_tensors="pt").to(self.device)
                    input_len = inputs["input_ids"].shape[-1]
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        logits_processor=[self.allowed_tokens_processor],
                    )
                    decoded_outputs = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
                else:
                    decoded_outputs = []
                    for choice in ref_col_choices:
                        messages = self.make_messages_with_choice(query, documents, choice)
                        output = self.llm.create_chat_completion(
                            messages=messages,
                            max_tokens=1,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            logits_processor=LogitsProcessorList([self.allowed_tokens_processor]),
                        )
                        decoded_outputs.append(output["choices"][0]["message"]["content"])
                res[ref_col] = [choice.split(":")[0] for i, choice in enumerate(ref_col_choices) if decoded_outputs[i].strip() == '1']
            else:
                # No choices: same flow as extract_possible_info_from_paper
                if self.use_hf:
                    prompt = self.make_prompt(query, documents)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                else:
                    messages = self.make_messages(query, documents)
                    response = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    response = response["choices"][0]["message"]["content"]
                res[ref_col] = self.extract_lst_from_llm_output(response)

        return res

    # def extract_possible_info_from_paper_with_choices(self, pmid: int, pmcid: str) -> Dict[str, List]:
    #     """
    #     Given a paper, extract all possible answer for each category, then based on the possible choice, harmonized them
    #     """
    #     res = {}

    #     for ref_col, ref_col_context, ref_col_choices in zip(self.referencing_col_lst, self.referencing_col_context_lst, self.referencing_col_choices_lst):
    #         # search for related context
    #         # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
    #         query = ref_col_context
    #         documents = self.vector_store.similarity_search_with_relevance_scores(
    #             query = query, 
    #             k = self.top_k,
    #             filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
    #         )
    #         documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
    #         # if no docs can found => no useful info 
    #         if len(documents) == 0:
    #             res[ref_col] = []
    #             continue

    #         # rerank
    #         scores = self.reranker_model.predict([(query, d) for d in documents])
    #         documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    #         documents = documents[:self.top_k_rerank]

    #         # extract a list of possible info from llm
    #         # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
    #         if self.use_hf:
    #             prompt = self.make_prompt(query, documents)
    #             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=self.max_new_tokens,
    #                 do_sample=False,
    #                 temperature=self.temperature,
    #                 top_p=self.top_p,
    #             )
    #             response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    #         else:
    #             messages = self.make_messages(query, documents)
    #             response = self.llm.create_chat_completion(
    #                 messages=messages,
    #                 max_tokens=self.max_new_tokens,
    #                 temperature=self.temperature,
    #                 top_p=self.top_p,
    #             )
    #             response = response["choices"][0]["message"]["content"]
    #         # prompt = self.make_prompt(full_query, documents)
    #         # outputs = self.model.generate(
    #         #     **self.tokenizer(prompt, return_tensors="pt").to(self.device),
    #         #     max_new_tokens=self.max_new_tokens,
    #         #     do_sample=False,
    #         #     temperature=self.temperature,
    #         #     top_p=self.top_p,
    #         # )
    #         # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         # extract list of possible info
    #         res[ref_col] = self.extract_lst_from_llm_output(response)

    #     print(res)
    #     for ref_col, ref_col_context, ref_col_choices in zip(self.referencing_col_lst, self.referencing_col_context_lst, self.referencing_col_choices_lst):
    #         # extra step
    #         if len(ref_col_choices) > 0 and len(res[ref_col]) > 0:
    #             res[ref_col] = self.calculate_best_choices(res[ref_col], ref_col_choices)
    
    #     return res

    # def extract_possible_info_from_paper_and_clues(self, pmid: int, pmcid: str, clues: List[str], batch_size: int = 16) -> Dict[str, Dict[str, List]]:
    #     """
    #     Given a paper, extract all possible answer for each category for each clue
    #     """
    #     res = {}

    #     for ref_col, ref_col_context in zip(self.referencing_col_lst, self.referencing_col_context_lst):
    #         # search for related context
    #         query = f"{ref_col_context}"
    #         documents = self.vector_store.similarity_search_with_relevance_scores(
    #             query = query, 
    #             k = self.top_k,
    #             filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
    #         )
    #         documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
    #         # if no docs can found => no useful info 
    #         if len(documents) == 0:
    #             res[ref_col] = []
    #             continue

    #         # rerank
    #         scores = self.reranker_model.predict([(query, d) for d in documents])
    #         documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    #         documents = documents[:self.top_k_rerank]

    #         # extract a list of possible info from llm
    #         full_query_lst = [f"What kind of {ref_col} is in the paper, given that {ref_col_context}, and clue {clue}" for clue in clues]
    #         if self.use_hf:
    #             prompts = [self.make_prompt(full_query, documents) for full_query in full_query_lst]
    #             clue_to_response = {}
    #             for i in range(0, len(prompts), batch_size):
    #                 prompts_batch = prompts[i: min(i + batch_size, len(prompts))]
    #                 outputs_batch = self.model.generate(
    #                     **self.tokenizer(prompts_batch, return_tensors="pt").to(self.device),
    #                     max_new_tokens=self.max_new_tokens,
    #                     do_sample=False,
    #                     temperature=self.temperature,
    #                     top_p=self.top_p,
    #                 )
    #                 responses_batch = self.tokenizer.batch_decode(outputs_batch, skip_special_tokens=True)
    #                 for j, response in enumerate(responses_batch):
    #                     clue_to_response[clues[i + j]] = self.extract_lst_from_llm_output(response)
    #         else:
    #             clue_to_response = {}
    #             for i, full_query in enumerate(full_query_lst):
    #                 messages = self.make_messages(full_query, documents)
    #                 response = self.llm.create_chat_completion(
    #                     messages=messages,
    #                     max_tokens=self.max_new_tokens,
    #                     temperature=self.temperature,
    #                     top_p=self.top_p,
    #                 )
    #                 response = response["choices"][0]["message"]["content"]
    #                 clue_to_response[clues[i]] = self.extract_lst_from_llm_output(response)
    #         # prompt = self.make_prompt(full_query, documents)
    #         # outputs = self.model.generate(
    #         #     **self.tokenizer(prompt, return_tensors="pt").to(self.device),
    #         #     max_new_tokens=self.max_new_tokens,
    #         #     do_sample=False,
    #         #     temperature=self.temperature,
    #         #     top_p=self.top_p,
    #         # )
    #         # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         # extract list of possible info
    #         res[ref_col] = deepcopy(clue_to_response)
    
    #     return res


def match_possible_info_to_df(df: pd.DataFrame, col_to_possible_info: Dict, 
                              embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizer,
                              threshold: float = 0.6):
    notes_col = [col for col in df.columns if "notes" in col]
    if len(notes_col) == 0:
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                df[col] = combine_possible_info(col_to_possible_info[col])
    else:
        # use info in notes to map, for each info, find the best one
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                used_col = []
                for n_col in notes_col:
                    # for each note, check if the match is actually related to that note by check the max similarity
                    unique_value = df[[n_col]].dropna()[n_col].unique().tolist()
                    similarity_score = calculate_similarity_scores(col_to_possible_info[col], unique_value, embeddings_model, embeddings_model_tokenizer) # #possible info * #unique value
                    # if torch.max(similarity_score) < 0.6:
                    # if best match do not have sim score at least 0.4 - 0.6
                    unique_value_to_possible_info = {}
                    if torch.min(torch.max(similarity_score, dim = 0).values) <= threshold:
                        for i, u in enumerate(unique_value):
                            # unique_value_to_possible_info[u] = combine_possible_info(col_to_possible_info[col])
                            unique_value_to_possible_info[u] = col_to_possible_info[col]
                    else:
                        # best_inx = torch.argmax(similarity_score, dim = 0)
                        # unique_value_to_possible_info = {}
                        # for i, u in enumerate(unique_value):
                        #     unique_value_to_possible_info[u] = col_to_possible_info[col][best_inx[i]]
                        for i, u in enumerate(unique_value):
                            valid_info = [col_to_possible_info[col][inx] for inx in range(similarity_score.shape[0]) if similarity_score[inx, i] >= threshold]
                            # unique_value_to_possible_info[u] = combine_possible_info(valid_info)
                            unique_value_to_possible_info[u] = valid_info
                    # for i, u in enumerate(unique_value):
                    #     for j, possible_value in enumerate(col_to_possible_info[col]):
                    #         if similarity_score[j, i] >= threshold:
                    #             if u not in unique_value_to_possible_info:
                    #                 unique_value_to_possible_info[u] = []
                    #             unique_value_to_possible_info[u].append(possible_value)
                    df[f"{col} from {n_col}"] = df[n_col].apply(lambda x: unique_value_to_possible_info.get(x, []))
                    used_col.append(f"{col} from {n_col}")
                df[col] = df[used_col].apply(lambda x: combine_possible_info(list(set(chain.from_iterable(x)))), axis = 1)
                df[col] = df[col].apply(lambda x: pd.NA if len(x) == 0 else x)
                df = df.drop(used_col, axis = 1)
    return df

def match_possible_info_to_df_with_clues(df: pd.DataFrame, pmid: int, pmcid: str, advp_information_retriever: ADVPInformationRetriever):
    notes_col = [col for col in df.columns if "notes" in col]
    if len(notes_col) == 0:
        col_to_possible_info = advp_information_retriever.extract_possible_info_from_paper(pmid, pmcid)
        for col in col_to_possible_info:
            if len(col_to_possible_info[col]) == 0:
                df[col] = pd.NA
            else:
                df[col] = combine_possible_info(col_to_possible_info[col])
    else:
        # use info in notes to map, for each info, find the best one
        col_to_used_col = {}
        for n_col in notes_col:
            clues = df[[n_col]].dropna()[n_col].unique().tolist()
            col_to_clues_to_possible_info = advp_information_retriever.extract_possible_info_from_paper_and_clues(pmid, pmcid, clues)
            for col in col_to_clues_to_possible_info:
                if col not in col_to_used_col:
                    col_to_used_col[col] = []
                df[f"{col} from {n_col}"] = df[n_col].apply(lambda x: col_to_clues_to_possible_info[col].get(x, []))
                col_to_used_col[col].append(f"{col} from {n_col}")
        for col in col_to_clues_to_possible_info:
            df[col] = df[col_to_used_col[col]].apply(lambda x: combine_possible_info_multilist(x), axis = 1)
            df[col] = df[col].apply(lambda x: pd.NA if len(x) == 0 else x)
            df = df.drop(col_to_used_col[col], axis = 1)
        return df