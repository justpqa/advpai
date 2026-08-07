from typing import List, Optional
import re
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

"""
This files is for general multi-purpose utils function
"""
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
    
def safe_int(x) -> Optional[int]:
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
    
def safe_round(x, num_digits) -> Optional[float]:
    try:
        return round(x, num_digits)
    except Exception:
        return None
    
def extract_first_number_from_str(s) -> Optional[int | float]:
    try:
        if isinstance(s, (int, float)):
            return s
        else:
            pattern = r"[\+\−]?\d+(?:\.\d+)?"
            match = re.search(pattern, s)
            if match:
                return match.group()
            else:
                return None
    except Exception:
        return None
    
def clean_snp(s) -> str:
    try:
        pattern = r"([rR][sS]\d+)"
        match = re.search(pattern, s)
        if match:
            return match.group()
        else:
            return None
    except Exception:
        return None
    
def make_embeddings(sentences: str | List[str], embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizer, normalize: bool = True) -> torch.Tensor:
    input = embeddings_model_tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # get token embeddings
    with torch.no_grad():
        output = embeddings_model(**input)
    token_embeddings = output[0]

    # extract mask and mean pooling for sentence embeddings
    input_mask_expanded = input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # final normalization
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    # size (# senetences, # dim)
    return embeddings

def calculate_similarity_scores(sentences_1: str | List[str], sentences_2: str | List[str], embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    embeddings_1, embeddings_2 = make_embeddings(sentences_1, embeddings_model, embeddings_model_tokenizer), make_embeddings(sentences_2, embeddings_model, embeddings_model_tokenizer)
    return embeddings_1 @ embeddings_2.T 

def calculate_similarity_scores_ip(sentences_1: str | List[str], sentences_2: str | List[str], embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    embeddings_1, embeddings_2 = make_embeddings(sentences_1, embeddings_model, embeddings_model_tokenizer, normalize = False), make_embeddings(sentences_2, embeddings_model, embeddings_model_tokenizer, normalize = False)
    return embeddings_1 @ embeddings_2.T 

def combine_possible_info(lst: List[str]):
    return " + ".join([x for x in list(set(lst)) if len(x) > 0])

def combine_possible_info_multilist(multilst: List[List[str]]):
    final_lst = []
    for lst in multilst:
        final_lst.extend(lst)
    return " + ".join([x for x in list(set(final_lst)) if len(x) > 0])