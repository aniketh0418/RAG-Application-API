from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pymongo import MongoClient
import os
import time
from model import process_document_and_answer

# from dotenv import load_dotenv
# load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
API_AUTHORIZED_TOKEN = os.getenv("API_AUTHORIZED_TOKEN")

app = FastAPI()

AUTHORIZED_TOKEN = API_AUTHORIZED_TOKEN

# --- MongoDB Setup ---
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
collection = db[MONGODB_COLLECTION_NAME]

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
        start_time = time.time()
        answers = process_document_and_answer(payload.documents, payload.questions)
        end_time = time.time()

        response_time_sec = round(end_time - start_time, 4)

        # --- Log to MongoDB ---
        now = datetime.now()
        doc_type = os.path.splitext(payload.documents.split("?")[0])[1].replace(".", "").lower()

        log_entry = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "doc_type": doc_type,
            "doc_url": payload.documents,
            "queries": payload.questions,
            "answers": answers,
            "response_time_seconds": response_time_sec
        }
        collection.insert_one(log_entry)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
