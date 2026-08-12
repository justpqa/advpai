import os
import requests
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
import warnings
warnings.filterwarnings("ignore")

CORPUS_PATHS = ["./papers/old", "./papers/test"]
PAPERS_INFO = [
    (30448613, "PMC6331247"), (30979435, "PMC6783343"), (28247064, "PMC5613285"), (30617256, "PMC6836675"),
    (30820047, "PMC6463297"), (29458411, "PMC5819208"), (29777097, "PMC5959890"), (30651383, "PMC6369905"),
    (28780673, "PMC5693762"), (30930738, "PMC6425305"), (31426376, "PMC6723529"), (29967939, "PMC6280657"),
    (29107063, "PMC5920782"), (29274321, "PMC5938137"), (30413934, "PMC6358498"), (30805717, "PMC7193309"),
    (30636644, "PMC6330399"), (29752348, "PMC5976227"), (28560309, "PMC5440281"), (27899424, "PMC5237405"),
    (31396565, 'PMC6677735'), (37198259, 'PMC10615750'), (37069360, 'PMC10115645'), (37349795, 'PMC10286470'), 
    (37621137, 'PMC10497850'), (35490390, 'PMC9622429'), (40666338, 'PMC12262740')
]
CHROMA_DB_PATH = "./chroma_db"
CHROMA_DB_COLLECTION_NAME = "gwas_paper_collection"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove tabs and newlines
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Remove special characters and punctuation (keep basic ones if needed)
    text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

# def ingest_corpus(corpus_path: str = CORPUS_PATH, chroma_db_path: str = CHROMA_DB_PATH,
#                   chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
#                   embedding_model_name: str = EMBEDDING_MODEL_NAME,
#                   chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, print_progress: bool = True) -> None:
#     # load documents from corpus
#     if print_progress:
#         print("Loading documents from corpus...")    
#     documents = []
#     metadata = [] # store a list of metadata dictionaries, currenly only have filenames
#     for filename in os.listdir(corpus_path):
#         pmid_pmcid = filename.split(".")[0]
#         pmid, pmcid = pmid_pmcid.split("_")
#         if filename.endswith(".txt"):
#             with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as f:
#                 documents.append(f.read())
#                 metadata.append({"PMID": pmid, "PMCID": pmcid})
#         elif filename.endswith(".pdf"):
#             pdf_path = os.path.join(corpus_path, filename)
#             with fitz.open(pdf_path) as doc:
#                 text = ""
#                 for page in doc:
#                     text += page.get_text() + "\n\n" # special indicator of pages
#                 documents.append(text)
#                 metadata.append({"PMID": pmid, "PMCID": pmcid})
#     if print_progress:
#         print(f"Finished loading {len(documents)} documents.")
#         print()

#     # split documents into chunks
#     if print_progress:
#         print("Splitting documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, 
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
#     splitted_documents = [clean_document(doc) for doc in splitted_documents]
#     if print_progress:
#         print(f"Finished splitting to make {len(splitted_documents)} chunks.")
#         print()

#     # create Chroma vector store
#     if print_progress:
#         print("Creating Chroma vector store...")
#     chroma_db = Chroma(
#         persist_directory=chroma_db_path,
#         embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
#         collection_name=chroma_db_collection_name,
#         collection_metadata={"hnsw:space": "cosine"}
#     )
#     # delete current collection contents before adding new documents
#     collection = chroma_db._collection
#     all_docs = collection.get(include=[])
#     all_ids = all_docs["ids"]
#     if all_ids:
#         collection.delete(ids=all_ids)
#     # add documents to Chroma vector store
#     chroma_db.add_documents(splitted_documents)
#     if print_progress:
#         print("Finished creating Chroma vector store.")
#         print()

def ingest_corpus_from_pmc(papers_info: List[Tuple[str, str]] = PAPERS_INFO, backup_corpus_path_lst: List[str] = CORPUS_PATHS,
                           chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME, 
                           embedding_model_name: str = EMBEDDING_MODEL_NAME, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, 
                           print_progress: bool = True):
    documents = []
    metadata = []
    for pmid, pmcid in papers_info:
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
        except Exception as e:
            print(f"Failed to extract paper {pmid}_{pmcid} from PMC with error {e}")
            for backup_corpus_path in backup_corpus_path_lst:
                if f"{pmid}_{pmcid}.txt" in os.listdir(backup_corpus_path):
                    with open(f"{backup_corpus_path}/{pmid}_{pmcid}.txt", 'r', encoding='utf-8') as f:
                        documents.append(f.read())
                        metadata.append({"PMID": str(pmid), "PMCID": pmcid})
                        break
                elif f"{pmid}_{pmcid}.pdf" in os.listdir(backup_corpus_path):
                    with fitz.open(f"{backup_corpus_path}/{pmid}_{pmcid}.pdf") as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text() + "\n\n" # special indicator of pages
                        documents.append(text)
                        metadata.append({"PMID": str(pmid), "PMCID": pmcid})
                        break
    if print_progress:
        print(f"Finished loading {len(documents)} documents.")
        print()

    # split documents into chunks
    if print_progress:
        print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
    splitted_documents = [clean_document(doc) for doc in splitted_documents]
    if print_progress:
        print(f"Finished splitting to make {len(splitted_documents)} chunks.")
        print()

    # create Chroma vector store
    if print_progress:
        print("Creating Chroma vector store...")
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
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
    if print_progress:
        print("Finished creating Chroma vector store.")
        print()
    

def verify_db_existence(chroma_db_path: str = CHROMA_DB_PATH,
                        chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME) -> bool:
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        collection_name=chroma_db_collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    return len(all_ids) > 0

if __name__ == "__main__":
    ingest_corpus_from_pmc()
    verify_db_existence()