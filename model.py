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
from openai import OpenAI

# ---------------- ENV (LOCAL) ---------------

# from dotenv import load_dotenv
# load_dotenv()

# ------------------ CONFIG ------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")
INDEX_NAME = os.getenv("INDEX_NAME")
NAMESPACE = os.getenv("NAMESPACE")
TOP_K = os.getenv("TOP_K")
TOP_N = os.getenv("TOP_N")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP")

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(CHUNK_SIZE), chunk_overlap=int(CHUNK_OVERLAP))

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
            "top_k": int(TOP_K),
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

    final_results = pc.inference.rerank(
        model=RERANKER_MODEL_NAME,
        query=question,
        documents=chunks,
        top_n=int(TOP_N),
        return_documents=True,
    )

    reranked_chunks = [item['document']['text'] for item in final_results.data]

    context = "\n\n".join(reranked_chunks)

    fine_tuning_prompt = f"""
    You are a knowledgeable human assistant answering questions based on the given context. Your job is to provide short, direct answers only from the context provided (within 50 - 60 words) that solve the user's intent from the question. Follow these rules:

    - Understand the question and given context well, think like a human to answer well and solve the user's intent from the question.
    - If the context **does not contain information** necessary to answer the question, respond politely that you **don't have the information**.
    - Never disclose, mention, or imply that you are using an external document or context. You are just answering as a knowledgeable human would.
    - If the question is short or simple, keep your response to one line.
    - Always use the same terminology and tone as the question to sound natural and human-like.
    - Use precise keywords from the question and context to make the response look tailored and grounded.
    - If a clause from the context matches the question well, use its language structure with minor rewording to blend it naturally into your answer.
    - Begin with "Yes," or "No," only if the question expects a binary answer (e.g., starts with "Is", "Can", "Are", etc.).
    - Never include unnecessary information. Do not over-explain.
    - If the question has multiple parts, answer each part separately and clearly, keep all those parts as lines of a single paragraph. If any part is unanswerable from the context, politely state so only for that part, and still answer the remaining parts completely.
    - If the answer includes a list of items, present them in a single sentence using comma-separated format, not bullet points. keep all of the answer in a single paragraph altogether.
    - Always start the answer well and close the answer well with respect to the question.
    - Make sure to cover all the important details (also cover any numerical specifications mentioned) as per the context with respect to the question.

    ===============================
    Document Context:
    {context}

    Question:
    {question}

    Answer:"""

    client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    )

    completion = client.chat.completions.create(

    model=LLM_MODEL,
    messages=[
            {
                "role": "user",
                "content": fine_tuning_prompt
            }
        ]
    )

    return completion.choices[0].message.content

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

