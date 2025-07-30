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
import google.generativeai as genai

# ------------------ CONFIG ------------------

PINECONE_API_KEY = "pcsk_28Ad9n_3ysjC12bv8Z91rEou1b9XiZ9RKb1QBF1n4uvKSF3Y9tCHPGJFy3utSi5CXcEDT9"
PINECONE_REGION = "us-east-1"
INDEX_NAME = "doc-index"
NAMESPACE = "docspace"
TOP_K = 15
TOP_N = 8

GOOGLE_API_KEY = "AIzaSyCS-fOKf6oYX9N8rSmzxRkbSPfsd6NXo_Q"
EMBED_MODEL_NAME = "llama-text-embed-v2"
RERANKER_MODEL_NAME = "bge-reranker-v2-m3"
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

    context = "\n\n".join(reranked_chunks)

    genai.configure(api_key=GOOGLE_API_KEY)

    generation_config = {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config,
#         system_instruction = f"""You are a concise and reliable assistant that answers insurance policy questions using only the provided document context.

# Respond in a formal, insurance-style tone using ONLY information from the retrieved context. Do not add assumptions or general knowledge.

# Word Limit Strategy:
# - If the answer is straightforward, answer in 15 - 20 words max.
# - If the answer includes multiple clauses or conditions, answer in 45 - 50 words max using commas or semicolons to stay brief.
# - Never exceed 50 words under any circumstance.

# Answer Rules:
# - Only begin the answer with “Yes,” or “No,” if the question clearly expects a binary response — i.e., starts with “Is”, “Does”, “Can”, “Are”, “Will”, etc.
# - Do NOT begin with “Yes,” or “No,” for questions that start with “What”, “When”, “How”, “Why”, “Where”, etc.
# - Be clear, precise, and legal-sounding.
# - Use structured insurance terms like “Sum Insured,” “waiting period,” “covered,” “excluded,” etc.
# - Do not repeat the question or include disclaimers.


#         Document Context:
#         {context}

#         Question:
#         {question}

#         Answer:"""

#     )

        system_instruction = f"""
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
- The answer must be a **single sentence or double sentanced** accord under 50 words.
- Never mention "Based on the context..." or "According to the document...".
- Prefer shorter clauses from the document instead of full paragraph-length sentences. Avoid excessive elaboration or restating policy names.

Style Preference:
- Use legal-style phrasing that mimics policy the text style present in the context.
- Maintain a neutral, natural, and professional tone.

===============================
Document Context:
{context}

Question:
{question}

Answer:""")



    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [f"{question}\n\n{context}"]}]
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"
    

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

## Example

# if __name__ == "__main__":

#     ## Request 1

    # blob_url = r"https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    # question_list = [
    # "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    # "What is the grace period for premium payment under the policy?",
    # "What is the waiting period for pre-existing diseases (PED) to be covered?",
    # "Does this policy cover maternity expenses, and what are the conditions?",
    # "What is the waiting period for cataract surgery?",
    # "Are the medical expenses for an organ donor covered under this policy?",
    # "What is the No Claim Discount (NCD) offered in this policy?",
    # "Is there a benefit for preventive health check-ups?",
    # "How does the policy define a 'Hospital'?",
    # "What is the extent of coverage for AYUSH treatments?",
    # "Are there any sub-limits on room rent and ICU charges for Plan A?"
    # ]

    ## Resuest 2

#     blob_url = r"https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
#     question_list = [
#     "What is the minimum hospitalization period required to make a claim?",
#     "How much is the room rent covered per day?",
#     "What is the waiting period for pre-existing diseases?",
#     "What is the pre-hospitalization coverage period?",
#     "What is the post-hospitalization coverage period?",
#     "What co-payment applies to claims for insured persons aged 75 or less?",
#     "Is AYUSH treatment covered under this policy?",
#     "What is the cataract treatment limit per eye?",
#     "Are maternity expenses covered?",
#     "What is the cumulative bonus for claim-free years?"
# ]


    # final_answers = process_document_and_answer(blob_url, question_list)
    # print("\nFinal Answer List:\n", final_answers)
    # # print(len(final_answers))
