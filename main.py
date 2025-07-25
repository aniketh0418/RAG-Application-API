from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pymongo import MongoClient
import os

from model import process_document_and_answer

app = FastAPI()

AUTHORIZED_TOKEN = "6a59087cab2c4d98677c22b8472cb4cac64d6968fea8507e9fcc2907a32f306a"

# --- MongoDB Setup ---
client = MongoClient("mongodb+srv://rapoluav12:OksS0OdkF4K6Lzrd@cluster1.ys8mpam.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["hackrx_db"]
collection = db["requests_log"]

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx_query(payload: QueryRequest, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization token format")

    token = authorization.replace("Bearer ", "").strip()

    if token != AUTHORIZED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or unauthorized token")

    try:
        answers = process_document_and_answer(payload.documents, payload.questions)

        # --- Log to MongoDB ---
        now = datetime.now()
        doc_type = os.path.splitext(payload.documents.split("?")[0])[1].replace(".", "").lower()

        log_entry = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "doc_type": doc_type,
            "doc_url": payload.documents,
            "queries": payload.questions,
            "answers": answers
        }
        collection.insert_one(log_entry)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))