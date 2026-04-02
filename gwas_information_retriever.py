from typing import Optional, List, Tuple, Dict
import re
import ast
import os
from copy import deepcopy
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor
from sentence_transformers import CrossEncoder
from llama_cpp import Llama
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

class AllowedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed_token_ids)] = 0
        return scores + mask
    
class GWASInformationRetriever:
    def __init__(self, referencing_col_df: pd.DataFrame, chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "gwas_paper_collection", 
                 embedding_model_name: str = "NeuML/pubmedbert-base-embeddings", reranker_model_name: str = "jinaai/jina-reranker-v1-turbo-en",
                 llm_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct", llm_gguf_path: Optional[str] = "./qwen2.5-7b-instruct-q4/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                 use_hf: bool = True, device: Optional[str] = None):
        # load ref col df
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        self.referencing_col_context_lst = referencing_col_df.apply(lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"], axis = 1).to_list()
        self.referencing_col_choices_lst = referencing_col_df["choices"].apply(lambda x: x.split(";") if isinstance(x, str) and ";" in x else []).to_list()

        # load vector store
        self.vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
            collection_name=chroma_db_collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
        # load the device
        self.device = device if device is not None else "cpu"

        # load the reranker
        self.reranker_model = CrossEncoder(
            model_name_or_path=reranker_model_name, 
            device = self.device,
            trust_remote_code = True
        ) 

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
        self.max_new_tokens = 1024
        self.similarity_score_threshold = 0.0
        self.temperature = 0
        self.top_p = 1
    
    def make_messages(self, query: str, documents: List[str]) -> List[Dict]:
        document_str = "\n\n".join([f"EXCERPT {i + 1}:\n{d}" for i, d in enumerate(documents)])
        # Example:
        # Question: What kind of study is in the paper? (Allowed values: "SNP-based", "gene-based")
        # Document: We performed both variant-level and gene-level association analyses. First, SNP-based GWAS summary statistics were computed in each ancestry group (EUR, EAS, AFR) and then combined via fixed-effect meta-analysis. In addition, we conducted a gene-based test aggregating rare variants per gene to prioritize candidate genes. The workflow included three stages: (i) discovery in EUR, (ii) validation in EAS and AFR, and (iii) trans-ancestry meta-analysis.
        # Output: ["gene-based", "SNP-based"]

        # Now do the same:

        messages = [
            {
                "role": "system", 
                "content": "You are a strict biomedical information extraction engine. Only use the provided text. Never guess. Output must follow the required format exactly."
            },
            {
                "role": "user",
                "content": f"""
Goal: extract candidate values for a field that will later be matched to table column headers.

Field:
{query}

Important:
- The output will be mapped to rows by matching against column names/headers.
- Therefore, prefer short “header-like” labels exactly as written (e.g., ADNI, IGAP, UK Biobank, discovery, replication).
- If both a long name and an abbreviation/alias appear, include BOTH as separate items (exact casing).
- If the field value applies to the entire paper (not tied to a specific cohort/stage), prefix it with GLOBAL: (e.g., GLOBAL: imputed to 1000 Genomes).

Evidence text:
{document_str}

Rules:
- Return exactly one Python list literal of strings and nothing else.
- Only include items explicitly supported by the excerpts.
- No paraphrasing; copy exact phrases/labels from the excerpts.
- De-duplicate items.
- If insufficient evidence, return [].

Output:"""
            }
        ]
        return messages

    def make_prompt(self, query: str, documents: List[str]) -> str:    
        messages = self.make_messages(query, documents)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    

#     def make_messages_with_choice(self, query: str, documents: List[str], choice: str) -> List[Dict]:
#         document_str = " ".join(documents).strip()

#         messages = [
#         {
#             "role": "user",
#             "content": f"""Check whether the document clearly supports the candidate answer.

# Question:
# {query}

# Candidate answer:
# {choice}

# Document:
# {document_str}

# Instructions:
# - Return 0 if the document clearly supports the candidate answer.
# - Return 1 if the document does not clearly support it.
# - Return only 0 or 1.
# - Do not explain.

# Example:
# Candidate answer: gene-based
# Document: We performed a gene-based test.
# Output: 0


# Output: """
#             }
#         ]
#         return messages

#     def make_prompt_lst_with_choices(self, query: str, documents: List[str], choices: List[str]) -> List[str]: 
#         prompt_lst = []
#         for choice in choices:
#             messages = self.make_messages_with_choice(query, documents, choice)
#             prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
#             prompt_lst.append(prompt)
#         return prompt_lst
    
    def extract_lst_from_llm_output(self, text: str) -> List[str]:
        text = text.replace("```", "").strip()
        matches = re.findall(r"\[.*?\]", text, re.DOTALL)
        if not matches or len(matches) < 1:
            return []
        list_str = matches[-1]
        try:
            return ast.literal_eval(list_str)
        except Exception:
            return []
        
    # def extract_lst_from_llm_output_choices(self, text: str) -> List[int]:
    #     # Remove code fences and trim
    #     text = text.replace("```", "").strip()

    #     # Extract the last bracketed list
    #     matches = re.findall(r"\[[^\]]*\]", text)
    #     if not matches:
    #         return []

    #     list_str = matches[-1]

    #     # Extract all integers inside the brackets
    #     numbers = re.findall(r"\d+", list_str)

    #     return [int(n) for n in numbers]

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

        for ref_col, ref_col_context in zip(self.referencing_col_lst, self.referencing_col_context_lst):
            # search for related context
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
            query = ref_col_context
            documents = self.vector_store.similarity_search_with_relevance_scores(
                query = query, 
                k = self.top_k,
                filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
            )
            documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
            # if no docs can found => no useful info 
            if len(documents) == 0:
                res[ref_col] = []
                continue

            # rerank
            scores = self.reranker_model.predict([(query, d) for d in documents])
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]

            # extract a list of possible info from llm
            # full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
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
            # prompt = self.make_prompt(full_query, documents)
            # outputs = self.model.generate(
            #     **self.tokenizer(prompt, return_tensors="pt").to(self.device),
            #     max_new_tokens=self.max_new_tokens,
            #     do_sample=False,
            #     temperature=self.temperature,
            #     top_p=self.top_p,
            # )
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # extract list of possible info
            res[ref_col] = self.extract_lst_from_llm_output(response)
    
        return res
    
    # def extract_possible_info_from_paper_with_choices(self, pmid: int, pmcid: str) -> Dict[str, List]:
    #     """
    #     Given a paper, extract all possible answer for each category
    #     """
    #     res = {}

    #     for ref_col, ref_col_context, ref_col_choices in zip(self.referencing_col_lst, self.referencing_col_context_lst, self.referencing_col_choices_lst):
    #         if len(ref_col_choices) > 0:
    #             query = f"{ref_col_context}"
    #             documents = self.vector_store.similarity_search_with_relevance_scores(
    #                 query = query, 
    #                 k = self.top_k,
    #                 filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
    #             )
    #             documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
    #             # if no docs can found => no useful info 
    #             if len(documents) == 0:
    #                 res[ref_col] = []
    #                 continue

    #             # rerank
    #             scores = self.reranker_model.predict([(query, d) for d in documents])
    #             documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    #             documents = documents[:self.top_k_rerank]

    #             # extract a list of possible info from llm
    #             full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
    #             prompt_lst = self.make_prompt_lst_with_choices(query, documents, ref_col_choices)
    #             outputs = self.model.generate(
    #                 **self.tokenizer(prompt_lst, padding=True, padding_side = "left", truncation=True, return_tensors="pt").to(self.device),
    #                 max_new_tokens=1,
    #                 do_sample=False,
    #                 temperature=self.temperature,
    #                 top_p=self.top_p,
    #                 logits_processor=[self.allowed_tokens_processor],
    #             )
    #             decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #             res[ref_col] = [choice for inx, choice in enumerate(ref_col_choices) if decoded_outputs[inx][-1] in ['0', '1']]
    #         else:
    #             # search for related context
    #             query = f"{ref_col_context}"
    #             documents = self.vector_store.similarity_search_with_relevance_scores(
    #                 query = query, 
    #                 k = self.top_k,
    #                 filter = {"$and": [{"PMID": str(pmid)}, {"PMCID": pmcid}]},
    #             )
    #             documents = [d.page_content for d, score in documents if score >= self.similarity_score_threshold]
    #             # if no docs can found => no useful info 
    #             if len(documents) == 0:
    #                 res[ref_col] = []
    #                 continue

    #             # rerank
    #             scores = self.reranker_model.predict([(query, d) for d in documents])
    #             documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    #             documents = documents[:self.top_k_rerank]

    #             # extract a list of possible info from llm
    #             full_query = f"What kind of {ref_col} is in the paper, given that {ref_col_context}"
    #             prompt = self.make_prompt(full_query, documents)
    #             # if self.use_hf:
    #             #     prompt = self.make_prompt(full_query, documents)
    #             #     outputs = self.model.generate(
    #             #         **self.tokenizer(prompt, return_tensors="pt").to(self.device),
    #             #         max_new_tokens=self.max_new_tokens,
    #             #         do_sample=False,
    #             #         temperature=self.temperature,
    #             #         top_p=self.top_p,
    #             #     )
    #             #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #             # else:
    #             #     messages = self.make_messages(full_query, documents)
    #             #     response = self.llm.create_chat_completion(
    #             #         messages=messages,
    #             #         max_tokens=self.max_new_tokens,
    #             #         temperature=self.temperature,
    #             #         top_p=self.top_p,
    #             #     )
    #             #     response = response["choices"][0]["message"]["content"]
    #             prompt = self.make_prompt(full_query, documents)
    #             outputs = self.model.generate(
    #                 **self.tokenizer(prompt, return_tensors="pt").to(self.device),
    #                 max_new_tokens=self.max_new_tokens,
    #                 do_sample=False,
    #                 temperature=self.temperature,
    #                 top_p=self.top_p,
    #             )
    #             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #             # extract list of possible info
    #             res[ref_col] = self.extract_lst_from_llm_output(response)
        
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