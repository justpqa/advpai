import re
import json
from copy import deepcopy
from typing import List, Union, Tuple, Dict
from collections.abc import Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
    
# class SingleTokenBiasProcessor(LogitsProcessor):
#     def __init__(self, token_ids, bias_value):
#         self.token_ids = token_ids
#         self.bias_value = bias_value

#     def __call__(self, input_ids, scores):
#         # Create a mask for allowed tokens
#         mask = torch.full_like(scores, -float("inf"))
#         for tid in self.token_ids:
#             mask[:, tid] = self.bias_value
#         return scores + mask
    
class ADVPFormattingEngine:
    def __init__(self, referencing_col_df: pd.DataFrame, embeddings_model_name: str = "NeuML/pubmedbert-base-embeddings"):
        # df of referencing col
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        self.referencing_col_context_lst = referencing_col_df.apply(lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"], axis = 1).to_list()
        self.referencing_col_example_lst = referencing_col_df["example"].to_list()
        self.referencing_col_group_lst = referencing_col_df["group"].to_list()
        self.referencing_col_with_multiple_copies = referencing_col_df[referencing_col_df["has_multiple_copies"]]["column"].tolist()
        # self.referencing_col_numeric = referencing_col_df[referencing_col_df["is_numeric"]]["column"].tolist()

        # Try to first convert any abbreviation, then we match column with right semantic
        self.advp_abbreviation_dict = {
            "CHR": "Chromosome number",
            "BP": "Base-pair position",
            "POS": "Position",
            "SNP": "Single nucleotide polymorphism identifier",
            "RS": "Reference Single nucleotide polymorphism",
            "VAR": "Variant",
            "ID": "identifier",
            "A1": "Effect allele / tested allele",
            "A2": "Other allele / non-effect allele",
            "REF": "Reference allele (genome reference)",
            "ALT": "Alternate allele",
            "EA": "Effect Allele",
            "NEA": "Non-Effect Allele",
            "RA": "Risk Allele",
            "OA": "Other Allele",
            "AF": "Allele Frequency (general term)",
            "RAF": "Risk Allele Frequency",
            "EAF": "Effect Allele Frequency",
            "MAF": "Minor Allele Frequency",
            "BETA": "Effect",
            "OR": "Odds Ratio",
            "SE": "Standard Error of effect estimate",
            "Z": "Z-score statistic",
            "T": "T-statistic",
            "CI": "Confidence Interval",
            "P": "P-value",
            "PVAL": "P-value",
            "LOGP": "Negative log10 P-value",
            "Q": "Heterogeneity statistic (meta-analysis)",
            "I2": "I-squared heterogeneity metric",
            "HWE": "Hardy-Weinberg Equilibrium test statistic",
            "INFO": "Imputation quality score",
            "R2": "Imputation accuracy metric",
            "CALLRATE": "Genotype call rate",
            "MISSING": "Missing genotype rate",
            "N": "Total sample size",
            "N_CASES": "Number of cases (for binary traits)",
            "N_CONTROLS": "Number of controls (for binary traits)",
            "EUR": "European ancestry",
            "AFR": "African ancestry",
            "ASN": "Asian ancestry",
            "AMR": "Admixed American ancestry",
            "SAS": "South Asian ancestry",
            "EAS": "East Asian ancestry",
            "LD": "Linkage Disequilibrium",
            "DPRIME": "LD D’ value",
            "CADD": "CADD score (functional impact)",
            "EQTL": "Expression quantitative trait locus",
            "PQTL": "Protein QTL",
            "GWGAS": "Gene-wide association study",
            "PRS": "Polygenic Risk Score",
            "PGS": "Polygenic Score",
            "QC": "Quality Control",
            "MA": "Meta-analysis",
            "HLA": "Human Leukocyte Antigen region",
            "HR": "Hazard ratio",
            "HET": "Heterogeneity test",
            "APOE4": "APOE ε4",
            "APOE*4": "APOE ε4",
            "#": "Number of",
            "frq": "Frequency",
            'β': "Effect",
            "nsnps": "Number of Variants"
        }

        # embeddings model
        self.embeddings_model_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
        self.embeddings_model = AutoModel.from_pretrained(embeddings_model_name)
        self.embeddings_model.eval()

        # also make col embeddings
        self.referencing_col_embeddings = self.create_col_embeddings_from_model(self.referencing_col_context_lst)
        self.referencing_col_example_embeddings = self.create_col_embeddings_from_model(self.referencing_col_example_lst)

        # NOTE: these are experimental part of using llm
        # adapter_path = "biomedlm_seqcls_lora"
        # base_model_name = "stanford-crfm/BioMedLM"
        # self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16, 
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        # base_model = AutoModelForSequenceClassification.from_pretrained(
        #     base_model_name,
        #     num_labels=4,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        # )
        # base_model.config.pad_token_id = self.tokenizer.pad_token_id
        # base_model.config.use_cache = False
        # self.model = PeftModel.from_pretrained(base_model, adapter_path)
        # self.model.eval()
        # self.device = "mps"

        # self.llm_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        # self.llm_model = AutoModelForCausalLM.from_pretrained(
        #     "Qwen/Qwen3.5-0.8B"
        # ).to("mps")
        # self.llm_model.eval()

        # # device
        # self.device = "mps"

    def clean_col(self, col: str) -> str:
        """
        Clean a column by replacing any possible abbreviation with their actual meaning for better semantic matching
        """
        new_col = col
        for abb in self.advp_abbreviation_dict:
            if re.search(fr"[^a-zA-Z]{abb.lower()}[^a-zA-Z]", new_col.lower()):
                new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}([^a-zA-Z])", fr"\1{self.advp_abbreviation_dict[abb]}\2", new_col.lower())
            elif re.search(fr"^{abb.lower()}[^a-zA-Z]", new_col.lower()):
                new_col = re.sub(fr"^{abb.lower()}([^a-zA-Z])", fr"{self.advp_abbreviation_dict[abb]}\1", new_col.lower())
            elif re.search(fr"[^a-zA-Z]{abb.lower()}$", new_col.lower()):
                new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}$", fr"\1{self.advp_abbreviation_dict[abb]}", new_col.lower())
            elif re.search(fr"^{abb.lower()}$", new_col.lower()):
                new_col = re.sub(fr"{abb.lower()}", self.advp_abbreviation_dict[abb], new_col.lower())
        # new_col = new_col.replace(".", " ")
        return new_col
    
    def make_col_prompt(self, col: str, example_values: Iterable, num_example_values: int = 5) -> str:
        """
        Based on the column title and some possible values of that columns, try to make a prompt
        """
        col_prompt = f"{col}: "
        col_example = f""
        for i in range(min(num_example_values, len(example_values))):
            col_prompt += f"{example_values[i]}, "
            col_example += f"{example_values[i]},"
        return col_prompt, col_example

    def create_col_embeddings_from_model(self, col: str | List[str]) -> np.ndarray | torch.Tensor:
        """
        Create embeddings from string represent col name or a prompt of that col
        """
        input = self.embeddings_model_tokenizer(col, padding=True, truncation=True, return_tensors='pt')

        # get token embeddings
        with torch.no_grad():
            output = self.embeddings_model(**input)
        token_embeddings = output[0]

        # extract mask and mean pooling for sentence embeddings
        input_mask_expanded = input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        col_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # final normalization
        col_embeddings = F.normalize(col_embeddings, p=2, dim=1)

        return col_embeddings
    
#     def make_prompt(self, col, candidates):
#         prompt = f"""Header: "{col}"
# Choose the best mapping:
# 1) {candidates[0]}
# 2) {candidates[1]}
# 3) {candidates[2]}
# Answer:"""
#         return prompt

#     def predict_class(self, prompt, max_length=256):
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=max_length,
#         ).to(self.device)

#         with torch.no_grad():
#             logits = self.model(**inputs).logits

#         return torch.argmax(logits, dim=-1).item()

#     def reranking_with_llm(self, col: str, candidates: List[str]) -> str:
#         """
#         LLM acts as a re-ranker to pick the best match from a list of candidates.
#         """
#         # Format the candidates as a numbered list for the LLM

#         prompt = f"""Task: Map clinical table headers to ADVP standard ontology, return a single number for the best choice, or 0 if nothing works

# Header: "p-value: 0.001, 5e-8, 0.43"
# Candidates: 
#     1. P-value: The statistical significance of the association. Keywords: P, P-value, P_adj, FDR. Examples: 5.0E-08, 0.0012, 1.2 x 10^-5, 0.05.
#     2. Effect Size: The magnitude and direction of the association. Keywords: Beta, OR, HR, Estimate. Examples: Beta=0.25, OR=1.45, HR=1.12, Log(OR)=0.37.
#     3. SNP: Variant identifier, or snp idenifier, or chr:pos. Keywords: chr:position, chr:pos, Variant, rsID, RS number, MarkerName, rs. Examples: rs12345, 20:45269867, 19:45411941:T:C, chr19:45411941, rs429358 (APOE ε4).
# Best Match: 1

# Header: "rs_number: rs123, rs456, rs789"
# Candidates: 
#     1. Chr: Genomic chromosome identifier. Keywords: CHR, Chrom, Chromosome. Examples: 1, 19, X, chr19, chrX.
#     2. Position: Genomic coordinate location. Keywords: BP, POS, Base Pair, start, end. Examples: 45411941, 10240500:10248600 (range), build 37.
#     3. SNP: Variant identifier, or snp idenifier, or chr:pos. Keywords: chr:position, chr:pos, Variant, rsID, RS number, MarkerName, rs. Examples: rs12345, 20:45269867, 19:45411941:T:C, chr19:45411941, rs429358 (APOE ε4).
# Best Match: 2

# Header: "Position: 40051948,71398604,150713594,50716490,58753575"
# Candidates: 
#     1. P-value: The statistical significance of the association. Keywords: P, P-value, P_adj, FDR. Examples: 5.0E-08, 0.0012, 1.2 x 10^-5, 0.05.
#     2. SNP: Variant identifier, or snp idenifier, or chr:pos. Keywords: chr:position, chr:pos, Variant, rsID, RS number, MarkerName, rs. Examples: rs12345, 20:45269867, 19:45411941:T:C, chr19:45411941, rs429358 (APOE ε4).
#     3. Cohort: The specific study or database name. Keywords: Study, Dataset, Discovery. Examples: ADNI, IGAP, UK Biobank, ADGC, CHARGE, EADI.
# Best Match: 0

# Header: "{col}"
# Candidates: 
#     - {candidates[0]}
#     - {candidates[1]}
#     - {candidates[2]}
# Best Match: """

#         allowed_indices = [str(i) for i in range(len(candidates) + 1)]
#         allowed_token_ids = [self.llm_model_tokenizer.encode(idx, add_special_tokens=False)[0] for idx in allowed_indices]
        
#         # logit bias to limit tokens that can be output
#         bias_processor = SingleTokenBiasProcessor(allowed_token_ids, 100.0)
#         logits_processor = LogitsProcessorList([bias_processor])

#         # 4. Generate exactly ONE token
#         inputs = self.llm_model_tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         with torch.no_grad():
#             output = self.llm_model.generate(
#                 **inputs,
#                 max_new_tokens=1,      # Force exactly one token
#                 logits_processor=logits_processor, # Force it to be one of our numbers
#                 pad_token_id=self.llm_model_tokenizer.eos_token_id,
#                 do_sample=False        # Greedy decoding for consistency
#             )

#         # 5. Extract and Convert to Integer
#         new_token = output[0][-1]
#         predicted_text = self.llm_model_tokenizer.decode(new_token).strip()
        
#         try:
#             idx = int(predicted_text) - 1 # Convert back to 0-based list index
#             return idx if idx >= 0 else None
#         except (ValueError, IndexError):
#             return None
        
    def match_single_col_to_ref_col(self, col: str, col_example: str) -> Tuple[str, float]:
        """
        match a single column to reference col given column, the list of referencing col and their embeddings
        """
        # calculate column embedding
        # col_embeddings = embeddings_model.encode(col, normalize_embeddings=True)
        # multi_index_pattern = r".+\..+"
        # if re.search(multi_index_pattern, col):
        #     col = col.replace(".", " ")
        #     # sub_col_lst = col.split(".")
        #     # sub_col_embeddings = create_embeddings_from_model(sub_col_lst, embeddings_model, embeddings_model_tokenizer)
        #     # col_embeddings = torch.mean(sub_col_embeddings, dim = 0)
        #     col = col.replace(".", " ")
        #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
        # else:
        #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
        col_embeddings = self.create_col_embeddings_from_model(col)
        col_embeddings = col_embeddings.reshape(-1, 1)
        col_example_embeddings = self.create_col_embeddings_from_model(col_example)
        col_example_embeddings = col_example_embeddings.reshape(-1, 1)

        # calculate similarity score
        scores = 1/2 * (torch.matmul(self.referencing_col_embeddings, col_embeddings).reshape(-1) + torch.matmul(self.referencing_col_example_embeddings, col_example_embeddings).reshape(-1))
        # scores = torch.matmul(self.referencing_col_embeddings, col_embeddings).reshape(-1)

        # sort similairty score
        top_k_indices = torch.argsort(scores, descending=True)

        # verify if we even got good enough similarity
        best_inx = top_k_indices[0].item()
        best_score = scores[best_inx].item()
        # second_best_inx = top_k_indices[1].item()
        # second_best_score = scores[second_best_inx].item()
        # need a threshold for score or else, just return col
        # if best_score < 0.4: 
        #     return (col, 1)
        
        # # now do rerank
        # candidates = [ref_col_lst[inx] for inx in top_k_indices if scores[inx] >= 0.4]
        # candidates_scores = [scores[inx] for inx in top_k_indices if scores[inx] >= 0.4]
        # best_ref_col, best_ref_col_scores = reranking_from_model(col, candidates, candidates_scores)
        # return (best_ref_col, best_ref_col_scores)

        # if best_score >= 0.4:
        #     return (self.referencing_col_lst[best_inx], best_score)
        # else:
        #     return (col, 1)
        
        threshold = 0.4 # NOTE: best after trying different val
        if best_score < threshold:
            return (col, float("inf"))
        else:
            candidates = [(self.referencing_col_lst[inx], scores[inx] - (self.referencing_col_group_lst[inx]) * 0) for inx in top_k_indices if scores[inx] >= threshold]
            candidates.sort(key = lambda x: x[1], reverse = True)
            return (candidates[0][0], candidates[0][1])

    # def match_single_col_to_ref_col(self, col: str, col_example: str) -> Tuple[str, float]:
    #     """
    #     match a single column to reference col given column, the list of referencing col and their embeddings
    #     """
    #     # calculate column embedding
    #     # col_embeddings = embeddings_model.encode(col, normalize_embeddings=True)
    #     # multi_index_pattern = r".+\..+"
    #     # if re.search(multi_index_pattern, col):
    #     #     col = col.replace(".", " ")
    #     #     # sub_col_lst = col.split(".")
    #     #     # sub_col_embeddings = create_embeddings_from_model(sub_col_lst, embeddings_model, embeddings_model_tokenizer)
    #     #     # col_embeddings = torch.mean(sub_col_embeddings, dim = 0)
    #     #     col = col.replace(".", " ")
    #     #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
    #     # else:
    #     #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
    #     col_embeddings = self.create_col_embeddings_from_model(col)
    #     col_embeddings = col_embeddings.reshape(-1, 1)
    #     col_example_embeddings = self.create_col_embeddings_from_model(col_example)
    #     col_example_embeddings = col_example_embeddings.reshape(-1, 1)

    #     # calculate similarity score
    #     scores = 1/2 * (torch.matmul(self.referencing_col_embeddings, col_embeddings).reshape(-1) + torch.matmul(self.referencing_col_example_embeddings, col_example_embeddings).reshape(-1))

    #     # sort similairty score
    #     top_k_indices = torch.argsort(scores, descending=True)

    #     # verify if we even got good enough similarity
    #     best_inx = top_k_indices[0].item()
    #     best_score = scores[best_inx].item()
    #     # second_best_inx = top_k_indices[1].item()
    #     # second_best_score = scores[second_best_inx].item()
    #     # need a threshold for score or else, just return col
    #     # if best_score < 0.4: 
    #     #     return (col, 1)
        
    #     # # now do rerank
    #     # candidates = [ref_col_lst[inx] for inx in top_k_indices if scores[inx] >= 0.4]
    #     # candidates_scores = [scores[inx] for inx in top_k_indices if scores[inx] >= 0.4]
    #     # best_ref_col, best_ref_col_scores = reranking_from_model(col, candidates, candidates_scores)
    #     # return (best_ref_col, best_ref_col_scores)

    #     if best_score >= 0.4:
    #         candidates_inx = []
    #         for i in range(3):
    #             candidates_inx.append((top_k_indices[i], scores[top_k_indices[i]]))
    #         return candidates_inx
    #     return []

    def match_many_col_to_ref_col(self, df: pd.DataFrame) -> Dict:
        """
        Given a list of column and a dataframe (could be dict later on if that fits better),
        try to match each column to the best fitted reference col, 
        return a dict of ref col : list of (col, cleaned col prompt, score)
        """

        # conduct matching
        multi_index_pattern_1 = r"^.+\..+$" # need multi index pattern for handling multi index
        multi_index_pattern_2 = r"^.+\|.+$"
        ref_col_to_col_lst = {}
        for col in df.columns:

            # extract values needed for prompt
            cleaned_col = self.clean_col(col)
            example_values = df[col].unique().tolist()
            # prompt: {col}: example, need to delete all : first

            if re.search(multi_index_pattern_1, cleaned_col.strip()):
                # try to assess each part and see if which one have highest score
                best_ref_col, best_score = None, 0
                best_cleaned_sub_col_prompt = None
                for sub_col in cleaned_col.split("."):

                    # make column prompt for each sub col
                    cleaned_sub_col_prompt, cleaned_sub_col_example = self.make_col_prompt(sub_col, example_values)
                    
                    # matching and compare
                    ref_col, score = self.match_single_col_to_ref_col(sub_col, cleaned_sub_col_example)
                    if score > best_score and score < float("inf"):
                        best_ref_col = ref_col
                        best_score = score
                
                # if we have a best one vs not
                if best_ref_col is not None:
                    if best_ref_col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[best_ref_col] = []
                    ref_col_to_col_lst[best_ref_col].append((col, best_score))
                else:
                    if col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[col] = []
                    ref_col_to_col_lst[col].append((col, float("inf")))
            elif re.search(multi_index_pattern_2, cleaned_col.strip()):
                # try to assess each part and see if which one have highest score
                best_ref_col, best_score = None, 0
                best_cleaned_sub_col_prompt = None
                for sub_col in cleaned_col.split("|"):

                    # make column prompt for each sub col
                    cleaned_sub_col_prompt, cleaned_sub_col_example = self.make_col_prompt(sub_col, example_values)
                    
                    # matching and compare
                    ref_col, score = self.match_single_col_to_ref_col(sub_col, cleaned_sub_col_example)
                    if score > best_score and score < float("inf"):
                        best_ref_col = ref_col
                        best_score = score
                
                # if we have a best one vs not
                if best_ref_col is not None:
                    if best_ref_col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[best_ref_col] = []
                    ref_col_to_col_lst[best_ref_col].append((col, best_score))
                else:
                    if col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[col] = []
                    ref_col_to_col_lst[col].append((col, float("inf")))
            else:
                # match a single col with best fit referencing col and return a dict
                # make the col prompt
                # extra steps to remove .
                cleaned_col = cleaned_col.replace(".", "").replace("|", "")
                cleaned_col_prompt, cleaned_col_example = self.make_col_prompt(cleaned_col, example_values)

                # matching for best col
                ref_col, score = self.match_single_col_to_ref_col(cleaned_col, cleaned_col_example)
                # if we still get same col
                if score == float("inf"): 
                    ref_col = col
                if ref_col not in ref_col_to_col_lst:
                    ref_col_to_col_lst[ref_col] = []
                ref_col_to_col_lst[ref_col].append((col, score))

        return ref_col_to_col_lst

    # def match_many_col_to_ref_col(self, df: pd.DataFrame) -> Dict:
    #     """
    #     Given a list of column and a dataframe (could be dict later on if that fits better),
    #     try to match each column to the best fitted reference col, 
    #     return a dict of ref col : list of (col, cleaned col prompt, score)
    #     """

    #     # conduct matching
    #     multi_index_pattern_1 = r"^.+\..+$" # need multi index pattern for handling multi index
    #     multi_index_pattern_2 = r"^.+\|.+$"
    #     ref_col_to_col_lst = {}
    #     for col in df.columns:

    #         # extract values needed for prompt
    #         cleaned_col = self.clean_col(col)
    #         example_values = df[col].unique().tolist()
    #         cleaned_col_prompt, cleaned_col_example = self.make_col_prompt(cleaned_col, example_values)
    #         # prompt: {col}: example, need to delete all : first

    #         if re.search(multi_index_pattern_1, cleaned_col.strip()):
    #             # try to assess each part and see if which one have highest score
    #             all_candidates = []
    #             for sub_col in cleaned_col.split("."):

    #                 # make column prompt for each sub col
    #                 cleaned_sub_col_prompt, cleaned_sub_col_example = self.make_col_prompt(sub_col, example_values)
                    
    #                 # matching and compare
    #                 candidates = self.match_single_col_to_ref_col(cleaned_sub_col_prompt, cleaned_sub_col_example)
    #                 all_candidates.extend(candidates)
                
    #             all_candidates.sort(key = lambda x: x[1], reverse = True)
    #             if len(all_candidates) >= 3:
    #                 candidates_ref_col = [self.referencing_col_lst[all_candidates[i][0]] for i in range(3)]
    #                 candidates_ref_col_score = [all_candidates[i][1] for i in range(3)]
    #                 candidates = [self.referencing_col_context_lst[all_candidates[i][0]] for i in range(3)]
    #                 ref_col_inx = self.reranking_with_llm(cleaned_col_prompt, candidates)
    #                 if ref_col_inx is None:
    #                     ref_col = col
    #                     ref_col_score = 1
    #                 else:
    #                     ref_col = candidates_ref_col[ref_col_inx]
    #                     ref_col_score = candidates_ref_col_score[ref_col_inx]
    #                 if ref_col not in ref_col_to_col_lst:
    #                     ref_col_to_col_lst[ref_col] = []
    #                 ref_col_to_col_lst[ref_col].append((col, ref_col_score))
    #             else:
    #                 ref_col_to_col_lst[col] = [(col, 1)]
    #         elif re.search(multi_index_pattern_2, cleaned_col.strip()):
    #             # try to assess each part and see if which one have highest score
    #             all_candidates = []
    #             for sub_col in cleaned_col.split("|"):

    #                 # make column prompt for each sub col
    #                 cleaned_sub_col_prompt, cleaned_sub_col_example = self.make_col_prompt(sub_col, example_values)
                    
    #                 # matching and compare
    #                 candidates = self.match_single_col_to_ref_col(cleaned_sub_col_prompt, cleaned_sub_col_example)
    #                 all_candidates.extend(candidates)
                
    #             all_candidates.sort(key = lambda x: x[1], reverse = True)
    #             if len(all_candidates) >= 3:
    #                 candidates_ref_col = [self.referencing_col_lst[all_candidates[i][0]] for i in range(3)]
    #                 candidates_ref_col_score = [all_candidates[i][1] for i in range(3)]
    #                 candidates = [self.referencing_col_context_lst[all_candidates[i][0]] for i in range(3)]
    #                 ref_col_inx = self.reranking_with_llm(cleaned_col_prompt, candidates)
    #                 if ref_col_inx is None:
    #                     ref_col = col
    #                     ref_col_score = 1
    #                 else:
    #                     ref_col = candidates_ref_col[ref_col_inx]
    #                     ref_col_score = candidates_ref_col_score[ref_col_inx]
    #                 if ref_col not in ref_col_to_col_lst:
    #                     ref_col_to_col_lst[ref_col] = []
    #                 ref_col_to_col_lst[ref_col].append((col, ref_col_score))
    #             else:
    #                 ref_col_to_col_lst[col] = [(col, 1)]
    #         else:
    #             # match a single col with best fit referencing col and return a dict
    #             # make the col prompt
    #             # extra steps to remove .
    #             cleaned_col = cleaned_col.replace(".", "")
    #             cleaned_col_prompt, cleaned_col_example = self.make_col_prompt(cleaned_col, example_values)

    #             # matching for best col
    #             candidates = self.match_single_col_to_ref_col(cleaned_col_prompt, cleaned_col_example)
    #             candidates.sort(key = lambda x: x[1], reverse = True)
    #             # if we still get same col
    #             if len(candidates) >= 3:
    #                 candidates_ref_col = [self.referencing_col_lst[candidates[i][0]] for i in range(3)]
    #                 candidates_ref_col_score = [candidates[i][1] for i in range(3)]
    #                 candidates = [self.referencing_col_context_lst[candidates[i][0]] for i in range(3)]
    #                 ref_col_inx = self.reranking_with_llm(cleaned_col_prompt, candidates)
    #                 if ref_col_inx is None:
    #                     ref_col = col
    #                     ref_col_score = 1
    #                 else:
    #                     ref_col = candidates_ref_col[ref_col_inx]
    #                     ref_col_score = candidates_ref_col_score[ref_col_inx]
    #                 if ref_col not in ref_col_to_col_lst:
    #                     ref_col_to_col_lst[ref_col] = []
    #                 ref_col_to_col_lst[ref_col].append((col, ref_col_score))
    #             else:
    #                 ref_col_to_col_lst[col] = [(col, 1)]
    #         # ref_col, score = match_single_col_to_ref_col(cleaned_col_prompt, ref_col_lst, ref_col_embeddings)
    #         # if ref_col == cleaned_col_prompt: 
    #         #     ref_col = col
    #         # if ref_col not in ref_col_to_col_lst:
    #         #     ref_col_to_col_lst[ref_col] = []
    #         # ref_col_to_col_lst[ref_col].append((col, cleaned_col_prompt, score))

    #     return ref_col_to_col_lst

    def format_original_table_from_col_matching(self, df: pd.DataFrame, new_col_to_old_col_lst: dict, remove_unique_col: bool = False):
        """
        Format final table by melt/make extra copies of reference columns, or simply rename column to reference column name
        """
        # possible_ref_col_to_melt = ["P-value", "Effect", "AF"]
        # map the columns
        # new_col_to_old_col_lst = advp_column_matching_engine.match_many_col_to_ref_col(df)
        df_with_ref_col = None
        new_col_to_not_melt = [] # list of columns that are stable and not need to be melt
        new_col_to_old_col_lst_to_melt = {}
        for new_col in new_col_to_old_col_lst:
            if len(new_col_to_old_col_lst[new_col]) == 1:
                # check if we remove unique col or not and if yes, is this a case of a col map to itself (score = 1)
                if (not remove_unique_col) or (remove_unique_col and new_col_to_old_col_lst[new_col][0][1] < float("inf")):
                    if df_with_ref_col is None:
                        df_with_ref_col = df[[new_col_to_old_col_lst[new_col][0][0]]]
                        df_with_ref_col = df_with_ref_col.rename({new_col_to_old_col_lst[new_col][0][0]: new_col}, axis = 1)
                    else:
                        df_with_ref_col[new_col] = df[new_col_to_old_col_lst[new_col][0][0]]
                    # add these single col to the list of not melt
                    new_col_to_not_melt.append(new_col)
            else:
                # if new_col in possible_ref_col_to_melt:
                #     old_col_lst = []
                #     for col, _ in new_col_to_old_col_lst[new_col]:
                #         # NOTE: special case, if col is the same as new_col, then melt will get error
                #         if col in new_col_to_old_col_lst:
                #             old_col_lst.append(col + " ")
                #         else:
                #             old_col_lst.append(col)
                #         if df_with_ref_col is None:
                #             if col in new_col_to_old_col_lst:
                #                 df_with_ref_col = df[[col]].rename({col: col + " "}, axis = 1)
                #             else:
                #                 df_with_ref_col = df[[col]]
                #         else:
                #             if col in new_col_to_old_col_lst:
                #                 df_with_ref_col[col + " "] = df[col]
                #             else:
                #                 df_with_ref_col[col] = df[col]
                #     new_col_to_old_col_lst_to_melt[new_col] = old_col_lst.copy()
                # else:
                #     # make multiple copies with notes
                #     for inx, (col, _) in enumerate(new_col_to_old_col_lst[new_col]):
                #         if df_with_ref_col is None:
                #             df_with_ref_col = df[[col]]
                #             df_with_ref_col = df_with_ref_col.rename({col: f"{new_col}_{inx + 1}"}, axis = 1)
                #         else:
                #             df_with_ref_col[f"{new_col}_{inx + 1}"] = df[col]
                #         df_with_ref_col[f"{new_col}_{inx + 1} notes"] = col
                #         # add these cols in group but not need to melt
                #         new_col_to_not_melt.append(f"{new_col}_{inx + 1}")
                #         new_col_to_not_melt.append(f"{new_col}_{inx + 1} notes")
                old_col_lst = []
                for col, _ in new_col_to_old_col_lst[new_col]:
                    # NOTE: special case, if col is the same as new_col, then melt will get error
                    if col in new_col_to_old_col_lst:
                        old_col_lst.append(col + " ")
                    else:
                        old_col_lst.append(col)
                    if df_with_ref_col is None:
                        if col in new_col_to_old_col_lst:
                            df_with_ref_col = df[[col]].rename({col: col + " "}, axis = 1)
                        else:
                            df_with_ref_col = df[[col]]
                    else:
                        if col in new_col_to_old_col_lst:
                            df_with_ref_col[col + " "] = df[col]
                        else:
                            df_with_ref_col[col] = df[col]
                new_col_to_old_col_lst_to_melt[new_col] = old_col_lst.copy()
        # Melting stage
        if len(new_col_to_old_col_lst_to_melt) > 0:
            # now melting column in same groups
            # Instead of keep melting, for each group, we make a new dataset of 
            # [stable col] + [to be melt col] => melt them as a new df
            # do this for each gorup and then join together based on stable col
            df_with_melt_col = None 
            for new_col in new_col_to_old_col_lst_to_melt:
                temp_df = deepcopy(df_with_ref_col[new_col_to_not_melt + new_col_to_old_col_lst_to_melt[new_col]])
                # create a temp row id for stable join
                temp_df["_row_id"] = np.arange(temp_df.shape[0])
                temp_df = temp_df.melt(
                    id_vars = new_col_to_not_melt + ["_row_id"],    
                    value_vars = new_col_to_old_col_lst_to_melt[new_col], 
                    var_name = f"{new_col} notes",      
                    value_name = f"{new_col}"
                )
                if df_with_melt_col is None:
                    df_with_melt_col = deepcopy(temp_df)
                else:
                    df_with_melt_col = df_with_melt_col.merge(temp_df, how = "inner", on = ["_row_id"] + new_col_to_not_melt)
            df_with_melt_col = df_with_melt_col.drop("_row_id", axis = 1)
            return df_with_melt_col
        else:
            # special case when all col are unique
            if df_with_ref_col is None:
                return pd.DataFrame()
            else:
                return df_with_ref_col

    def format_original_table(self, df: pd.DataFrame, remove_unique_col: bool = False):
        """
        Format final table by running pipeline of col matching -> final format
        """
        # first do matching
        ref_col_to_col_lst = self.match_many_col_to_ref_col(df)

        # finally, filter col that cannot have multiple copies
        for ref_col in ref_col_to_col_lst:
            if ref_col not in self.referencing_col_with_multiple_copies and len(ref_col_to_col_lst[ref_col]) > 1:
                best_col, best_score = None, float("-inf")
                for col, score in ref_col_to_col_lst[ref_col]:
                    if score > best_score:
                        best_col = col
                        best_score = score
                ref_col_to_col_lst[ref_col] = [(best_col, best_score)]

        # then do final melt
        final_df = self.format_original_table_from_col_matching(df, ref_col_to_col_lst, remove_unique_col)

        # columns that is not in referencing col list will be created later
        for col in self.referencing_col_lst:
            if col not in final_df.columns:
                final_df[col] = pd.NA
        return final_df