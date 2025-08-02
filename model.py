import os
import requests
import tempfile
import hashlib
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDoc
from docx import Document as DocxDocument
import mailparser
from pinecone import Pinecone
from mistralai import Mistral

# ------------------ CONFIG ------------------

PINECONE_API_KEY = "pcsk_28Ad9n_3ysjC12bv8Z91rEou1b9XiZ9RKb1QBF1n4uvKSF3Y9tCHPGJFy3utSi5CXcEDT9"
PINECONE_REGION = "us-east-1"
INDEX_NAME = "doc-index"
NAMESPACE = "docspace"
TOP_K = 8
# TOP_N = 8
LLM_API_KEY = "Xum09rLjdkRxsBN67AyzfO4eVGaiaSxF"
LLM_MODEL = "mistral-large-latest"
EMBED_MODEL_NAME = "llama-text-embed-v2"
# RERANKER_MODEL_NAME = "pinecone-rerank-v0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region=PINECONE_REGION,
        embed={
            "model": EMBED_MODEL_NAME,
            "field_map": {"text": "chunk_text"}
        }
    )

index = pc.Index(INDEX_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ------------------ UTILS ------------------

def get_doc_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def extract_text_from_docx(file_path: str) -> List[LangchainDoc]:
    doc = DocxDocument(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [LangchainDoc(page_content=text)]


def extract_text_from_eml(file_path: str) -> List[LangchainDoc]:
    parsed = mailparser.parse_from_file(file_path)
    body = parsed.body
    return [LangchainDoc(page_content=body)]    

def fetch_pdf_from_blob_url(blob_url: str) -> str:
    response = requests.get(blob_url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF from Blob URL")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def check_if_doc_exists(blob_url: str, vector_dim: int = 1024) -> bool:
    response = index.query(
        vector=[0.0] * vector_dim,
        top_k=1,
        namespace=NAMESPACE,
        filter={"blob_url": {"$eq": blob_url}}
    )
    return len(response.get("matches", [])) > 0

def embed_and_store(file_path: str, doc_id: str, blob_url: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == ".docx":
        docs = extract_text_from_docx(file_path)
    elif ext == ".eml":
        docs = extract_text_from_eml(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    chunks = text_splitter.split_documents(docs)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"{doc_id}_{i}",
            "chunk_text": chunk.page_content,
            "doc_id": str(doc_id),
            "blob_url": str(blob_url),
            "chunk_index": str(i)
        })

    BATCH_SIZE = 96
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert_records(namespace=NAMESPACE, records=batch)

def retrieve_and_answer(question: str, blob_url: str):
    results = index.search(
        namespace=NAMESPACE,
        query={
            "top_k": TOP_K,
            "inputs": {
                "text": question
            },
            "filter": {
                "blob_url": blob_url
            }
        }
    )

    hits = results.get("result", {}).get("hits", [])

    if not hits:
        return "No relevant context found to answer the question."
    
    chunks = [hit.get("fields", {}).get("chunk_text", "") for hit in hits]

    # final_results = pc.inference.rerank(
    #     model=RERANKER_MODEL_NAME,
    #     query=question,
    #     documents=chunks,
    #     top_n=TOP_N,
    #     return_documents=True,
    # )

    # reranked_chunks = [item['document']['text'] for item in final_results.data]

    context = "\n\n".join(chunks)

    client = Mistral(api_key=LLM_API_KEY)

    fine_tuning_prompt = f"""
    You are a human assistant in the backend, provided is the context and question that you must answer within 50 to 60 words (2 lines) maximum depending on the intent required to cover. If it looks like a simple and short question that does not require long answer keep it one-lined. Make sure to cover all the important details (also cover any numerical specifications mentioned) as per the context with respect to the question. MAKE SURE THE ANSWER'S INTENT IS COMPLETELY SOLVING THE QUESTION. Use the language and wording format present in the question with respect to the context, just in the way of how a human would have do it, if he was given a document and questions to answer. Humanize your answer. Complete your answer well, close the answer well according to the question. If the context contains a policy clause that answers the question, extract and closely imitate that clause with some rehprasing in order to present the answer in the question's tone. Begin with “Yes,” or “No,” only if the question expects a binary response (i.e., starts with “Is”, “Does”, “Can”, “Are”, etc.). Do NOT begin with “Yes,” or “No,” for descriptive questions (i.e., “What”, “When”, “How”, etc.). Try to maintain simple language. Minimal rephrasing is allowed. Do NOT disclose that you are getting the context from a document in any manner and if any question asked is out of context, responsed politely that you can not answer it because it is out of your context or knowledge - remember you are like a human in the backend, who does not point out to any document or explicit context. 

    ===============================
    Document Context:
    {context}

    Question:
    {question}

    Answer:"""

    while True:
        try:
            messages = [
                {
                    "role": "user", "content": fine_tuning_prompt
                }
            ]
            chat_response = client.chat.complete(
                model=LLM_MODEL,
                messages=messages
            )
            return (chat_response.choices[0].message.content)
        except Exception as e:
            continue
    

# ------------------ MAIN FUNCTION ------------------

def process_document_and_answer(blob_url: str, questions: List[str]) -> List[str]:
    doc_id = get_doc_id_from_url(blob_url)

    if not check_if_doc_exists(blob_url):
        temp_path = fetch_pdf_from_blob_url(blob_url)
        embed_and_store(temp_path, doc_id, blob_url)
        os.remove(temp_path)

        for _ in range(10):
            time.sleep(1)
            if check_if_doc_exists(doc_id):
                break

    answers = []
    for q in questions:
        ans = retrieve_and_answer(q, blob_url)
        answers.append(ans)

    return answers
