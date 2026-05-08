from typing import Optional, List, Tuple, Dict
import re
import ast
import os
from copy import deepcopy
import requests
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, AutoModel
from sentence_transformers import CrossEncoder
from llama_cpp import Llama, LogitsProcessorList
from huggingface_hub import login
from utils import *
import langextract as lx
from langextract.providers import ollama
from dotenv import load_dotenv
load_dotenv()

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
    
class GWASInformationRetriever:
    def __init__(self, referencing_col_df: pd.DataFrame, referencing_col_with_choice_df: Optional[pd.DataFrame] = None, 
                 chroma_db_path: str = "./chroma_db", chroma_db_collection_name: str = "gwas_paper_collection", 
                 embeddings_model_name: str = "NeuML/pubmedbert-base-embeddings", reranker_model_name: str = "BAAI/bge-reranker-base",
                 llm_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct", llm_gguf_path: Optional[str] = "./qwen2.5-3b-instruct-q8/qwen2.5-3b-instruct-q8_0.gguf",
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
            embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_name),
            collection_name=chroma_db_collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
        # load the device
        self.device = device if device is not None else "cpu"
        
        # load the embeddings
        # self.embeddings_model_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
        # self.embeddings_model = AutoModel.from_pretrained(
        #     embeddings_model_name,
        #     device_map = self.device,
        # )

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

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

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
            scores = self.reranker_model.predict([(retrieval_query, d) for d in documents])
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:self.top_k_rerank]
            for doc in documents:
                messages = self.make_messages(query, [doc], examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
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
                response = response["choices"][0]["message"]["content"]
                if ref_col not in res:
                    res[ref_col] = []
                new_info = self.extract_lst_from_llm_output(response)
                new_info = list(map(lambda x: x.lower(), new_info))
                res[ref_col] = list(set(res[ref_col] + new_info))

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


class GWASInformationRetrieverKeyword:
    def __init__(self, info_type: str, keyword_dict: Dict[str, List]):
        if keyword_dict is None:
            raise Exception("Please include a keyword dictionary")
        self.info_type = info_type
        self.keyword_dict = keyword_dict
    
    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> List[str]:
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
        possible_info = []
        for keyword in self.keyword_dict:
            for keyword_variation in self.keyword_dict[keyword]:
                if keyword_variation in curr_doc:
                    print(keyword, keyword_variation)
                    possible_info.append(keyword)
                    break
        return possible_info
    
class GWASInformationRetrieverV2:
    def __init__(self, referencing_col_df: pd.DataFrame, reranker_model_name: str = "BAAI/bge-reranker-base",
                 llm_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct", llm_gguf_path: Optional[str] = "./qwen2.5-3b-instruct-q8/qwen2.5-3b-instruct-q8_0.gguf",
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
        # load keyword
        self.referencing_col_keywords_lst = [
            [k.strip() for k in s.split(";") if k.strip()]
            for s in referencing_col_df["search_keywords"]
        ]

        # load the device
        self.device = device if device is not None else "cpu"

        # load the reranker
        self.reranker_model = CrossEncoder(
            model_name_or_path=reranker_model_name, 
            device = self.device,
            trust_remote_code = True
        ) 

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

        # NOTE: config for search and generate, add it as params later
        self.top_k = 20
        self.top_k_rerank = 5
        self.max_new_tokens = 128
        self.similarity_score_threshold = 0.0
        self.temperature = 0
        self.top_p = 1
    
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

    def _get_chunk_span_around_position(
        self,
        text: str,
        match_start: int,
        match_end: int,
        min_tokens: int = 450,
        max_tokens: int = 550,
        target_tokens: int = 500,
    ) -> Tuple[int, int, str]:
        """
        Expand outward from a keyword hit through paragraph > sentence > word
        boundaries. Returns (start_idx, end_idx, chunk_text).
        Approx 1 token ~= 4 chars.
        """
        approx_char_budget = target_tokens * 4
        half = approx_char_budget // 2
        left = max(0, match_start - half)
        right = min(len(text), match_end + half)

        def snap_left(pos: int, sep: str) -> int:
            matches = list(re.finditer(sep, text[:pos]))
            return matches[-1].end() if matches else 0

        def snap_right(pos: int, sep: str) -> int:
            m = re.search(sep, text[pos:])
            return pos + m.start() if m else len(text)

        separators = [r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"]

        for i, sep in enumerate(separators):
            new_left = snap_left(left, sep)
            new_right = snap_right(right, sep)
            chunk = text[new_left:new_right].strip()
            n_tokens = len(chunk) // 4
            if min_tokens <= n_tokens <= max_tokens:
                return new_left, new_right, chunk
            if i == 0 and n_tokens < min_tokens:
                new_left = snap_left(max(0, new_left - 1), sep)
                new_right = snap_right(min(len(text), new_right + 1), sep)
                chunk = text[new_left:new_right].strip()
                if len(chunk) // 4 <= max_tokens:
                    return new_left, new_right, chunk

        left = snap_left(left, r"\s+")
        right = snap_right(right, r"\s+")
        return left, right, text[left:right].strip()

    def get_chunk_around_position(
        self,
        text: str,
        match_start: int,
        match_end: int,
        min_tokens: int = 450,
        max_tokens: int = 550,
        target_tokens: int = 500,
    ) -> str:
        _, _, chunk = self._get_chunk_span_around_position(
            text, match_start, match_end,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            target_tokens=target_tokens,
        )
        return chunk

    def get_documents_by_keyword(
        self,
        text: str,
        keyword: str,
        min_tokens: int = 450,
        max_tokens: int = 550,
        target_tokens: int = 500,
    ) -> List[str]:
        """
        Find all occurrences of `keyword` (case-insensitive, word-boundary aware
        for alphabetic keywords) and return a coherent chunk around each.
        Overlapping chunks are merged.
        """
        if not keyword:
            return []

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\s\-]*", keyword):
            pattern = r"\b" + re.escape(keyword) + r"\b"
        else:
            pattern = re.escape(keyword)

        spans = []
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
        if not spans:
            return []

        # Track chunk spans directly from the snapping logic (case-independent).
        chunks = []
        for s, e in spans:
            start_idx, end_idx, chunk = self._get_chunk_span_around_position(
                text, s, e,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                target_tokens=target_tokens,
            )
            chunks.append((start_idx, end_idx, chunk))

        # Merge overlapping chunks by span
        chunks.sort(key=lambda x: x[0])
        merged = []
        for start_idx, end_idx, chunk in chunks:
            if merged and start_idx <= merged[-1][1]:
                prev_s, prev_e, _ = merged[-1]
                new_e = max(prev_e, end_idx)
                merged[-1] = (prev_s, new_e, text[prev_s:new_e].strip())
            else:
                merged.append((start_idx, end_idx, chunk))
        return [c for _, _, c in merged if c]

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        """
        Given a paper, extract all possible answer for each category
        """
        res = {}

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

        for ref_col, ref_col_context, ref_col_examples, ref_col_use_examples_in_llm, ref_col_retrieval_query, ref_col_keywords in zip(
            self.referencing_col_lst,
            self.referencing_col_context_lst,
            self.referencing_col_examples_lst,
            self.referencing_col_use_examples_in_llm_lst,
            self.referencing_col_retrieval_query_lst,
            self.referencing_col_keywords_lst
        ):
            # Retrieval uses definition + examples (better recall); the LLM
            # prompt sees only the definition, with examples in a quarantined block.
            query = ref_col_context
            retrieval_query = ref_col_retrieval_query
            
            documents = []
            seen = set()
            for keyword in ref_col_keywords:
                for chunk in self.get_documents_by_keyword(curr_doc, keyword):
                    key = chunk.lower()
                    if key not in seen:
                        seen.add(key)
                        documents.append(chunk)

            if not documents:
                res[ref_col] = []
                continue

            # rerank
            scores = self.reranker_model.predict([(retrieval_query, d) for d in documents])
            documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            documents = documents[:min(self.top_k_rerank, len(documents))]

            messages = self.make_messages(query, documents, examples=ref_col_examples, use_examples_in_llm=ref_col_use_examples_in_llm)
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
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
            response = response["choices"][0]["message"]["content"]
            res[ref_col] = self.extract_lst_from_llm_output(response)

        return res

class GWASInformationRetrieverV3:
    def __init__(self, referencing_col_df: pd.DataFrame, refencing_col_example_dict_lst: List[Dict],
                 ollama_model_id: str = "qwen2.5-3b-instruct-q8"):
        # col lst
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        # context lst
        self.referencing_col_context_lst = referencing_col_df.apply(
            lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"],
            axis=1,
        ).to_list()

        # set up examples
        self.ref_col_examples = []
        for example_dict in refencing_col_example_dict_lst:
            extraction_lst = []
            for key in example_dict:
                if key in self.referencing_col_lst:
                    for info in example_dict[key]:
                        extraction_lst.append(
                            lx.data.Extraction(extraction_class=key, extraction_text=info)
                        )
            self.ref_col_examples.append(
                lx.data.ExampleData(text = example_dict["text"], extractions=extraction_lst)
            )
        
        self.ollama_model_id = ollama_model_id

    def extract_possible_info_from_paper(self, pmid: int, pmcid: str) -> Dict[str, List]:
        # get text
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

        # prompt 
        referencing_col_context_str = "\n".join([f"-{c}" for c in self.referencing_col_context_lst])
        prompt = f"""Extract these info:\n{referencing_col_context_str}"""

        # extract
        result = lx.extract(
            text_or_documents=curr_doc,
            prompt_description=prompt,
            examples=self.ref_col_examples,
            model_id=self.ollama_model_id,                 # your ollama model
            model_url="http://localhost:11434",
            max_char_buffer=500,
            context_window_chars=100,
            temperature=0,
            resolver_params={
                "format_handler": ollama.OLLAMA_FORMAT_HANDLER
            },
        )
        res = {}
        for col in self.referencing_col_lst:
            res[col] = []
        for extract in result.extractions:
            if extract.extraction_class not in res:
                res[extract.extraction_class] = []
            extracted_text = extract.extraction_text.lower()
            if extracted_text not in res[extract.extraction_class]:
                res[extract.extraction_class].append(extracted_text)
        return res

        

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
                            unique_value_to_possible_info[u] = combine_possible_info(col_to_possible_info[col])
                    else:
                        # best_inx = torch.argmax(similarity_score, dim = 0)
                        # unique_value_to_possible_info = {}
                        # for i, u in enumerate(unique_value):
                        #     unique_value_to_possible_info[u] = col_to_possible_info[col][best_inx[i]]
                        for i, u in enumerate(unique_value):
                            valid_info = [col_to_possible_info[col][inx] for inx in range(similarity_score.shape[0]) if similarity_score[inx, i] >= threshold]
                            unique_value_to_possible_info[u] = combine_possible_info(valid_info)
                    df[f"{col} from {n_col}"] = df[n_col].apply(lambda x: unique_value_to_possible_info.get(x, ""))
                    used_col.append(f"{col} from {n_col}")
                df[col] = df[used_col].apply(lambda x: combine_possible_info(x), axis = 1)
                df[col] = df[col].apply(lambda x: pd.NA if len(x) == 0 else x)
                df = df.drop(used_col, axis = 1)
    return df

def match_possible_info_to_df_with_clues(df: pd.DataFrame, pmid: int, pmcid: str, gwas_information_retriever: GWASInformationRetriever):
    notes_col = [col for col in df.columns if "notes" in col]
    if len(notes_col) == 0:
        col_to_possible_info = gwas_information_retriever.extract_possible_info_from_paper(pmid, pmcid)
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
            col_to_clues_to_possible_info = gwas_information_retriever.extract_possible_info_from_paper_and_clues(pmid, pmcid, clues)
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