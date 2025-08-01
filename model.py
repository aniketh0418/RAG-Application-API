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
TOP_N = 8
LLM_API_KEY = "Xum09rLjdkRxsBN67AyzfO4eVGaiaSxF"
LLM_MODEL = "mistral-large-latest"
EMBED_MODEL_NAME = "llama-text-embed-v2"
RERANKER_MODEL_NAME = "pinecone-rerank-v0"
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

def check_if_doc_exists(doc_id: str) -> bool:
    stats = index.describe_index_stats()
    return doc_id in stats.get("namespaces", {})

def embed_and_store(file_path: str, doc_id: str):
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
            "doc_id": doc_id
        })

    BATCH_SIZE = 96
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert_records(namespace=doc_id, records=batch)

def retrieve_and_answer(question: str, namespace: str):
    results = index.search(
        namespace=namespace,
        query={
            "top_k": TOP_K,
            "inputs": {
                "text": question
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
        top_n=TOP_N,
        return_documents=True,
    )

    reranked_chunks = [item['document']['text'] for item in final_results.data]

    context = "\n\n".join(chunks)

    client = Mistral(api_key=LLM_API_KEY)

    fine_tuning_prompt = f"""
    You are a formal and precise assistant (like a human in the backend) that answers insurance policy questions strictly using the provided document context.

    Objective:
    Craft answers that are concise, accurate, and match the phrasing and tone typically found in insurance policy documents. Avoid paraphrasing — prefer **verbatim extraction** from context whenever possible.

    Answer Formatting Rules:
    - Respond using complete sentences that sound like policy clauses or contractual statements.
    - If the context contains a policy clause that answers the question, extract or closely imitate that clause.
    - Begin with “Yes,” or “No,” only if the question expects a binary response (i.e., starts with “Is”, “Does”, “Can”, “Are”, etc.).
    - Do NOT begin with “Yes,” or “No,” for descriptive questions (i.e., “What”, “When”, “How”, etc.).
    - Use domain-specific terms such as “Sum Insured”, “Grace Period”, “waiting period”, “excluded”, “indemnified”, etc., according to the question and context.

    Strict Output Rules:
    - Do NOT make assumptions or include general knowledge.
    - Do NOT include disclaimers, question restatement, or introductory phrases.
    - End every sentence with a **period**.
    - The answer must be a **single sentence or double sentanced** accord under 50 words, if more important details are to be covered - keep it under 60 words.
    - Never mention "Based on the context..." or "According to the document...".
    - Prefer shorter clauses from the document instead of full paragraph-length sentences. Avoid excessive elaboration or restating policy names.
    - Maintain a human-like tone—**never reveal that the answer is coming from a document**.
    - Include all the required **key-words** in the answer, with respect to the question and context.

    Style Preference:
    - Use legal-style phrasing that mimics policy the text style present in the context.
    - Maintain a neutral, natural, and professional tone.
    - Strictly maintain simple language, don't give complicative answer that is not easy to understand on a single read with respect to the question.

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

import time

def process_document_and_answer(blob_url: str, questions: List[str]) -> List[str]:
    doc_id = get_doc_id_from_url(blob_url)

    if not check_if_doc_exists(doc_id):
        # print("New Doc!")
        file_ext = os.path.splitext(blob_url.split("?")[0])[1].lower()
        if file_ext not in [".pdf", ".docx", ".eml"]:
            raise ValueError("Unsupported file type.")
        temp_path = fetch_pdf_from_blob_url(blob_url)
        embed_and_store(temp_path, doc_id)
        os.remove(temp_path)

        for _ in range(5):
            time.sleep(1)
            if check_if_doc_exists(doc_id):
                break

    answers = []
    for q in questions:
        ans = retrieve_and_answer(q, doc_id)
        answers.append(ans)

    return answers
