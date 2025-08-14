# RAG Application API

A Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, Pinecone, and OpenAI to process documents from a given URL and answer user questions based on their content. The API handles PDFs, stores their embeddings in Pinecone, retrieves relevant chunks, and generates concise, context-grounded answers.

---

# ✨ Features

- **Blob URL based:** Blob PDF Documents.
- **Automatic duplicate detection** — avoids re-embedding existing docs, with reference to meta data.
- **Retrieval-Augmented Generation pipeline** using Pinecone vector database.
- **Contextual answer generation** via OpenAI LLMs.
- **Embedding and Reranking** with Pinecone's inference API for more relevant context. For effecient embedding and reranking use `llama-text-embed-v2` and `bge-reranker-v2-m3` respectively.
- **FastAPI-powered** RESTful endpoint.
- **MongoDB** logging for all requests and responses.
- **Bearer token authentication** for secure access.

---

# ⚙️ How It Works

### 1️⃣ Client sends request → `/hackrx/run` endpoint with:

- A document URL
- A list of questions
- Authorization header with Bearer token

### 3️⃣ Document processing:

- Checks if document already exists in Pinecone (via `blob_url`)
- If not, downloads and extracts text
- Splits into chunks (`RecursiveCharacterTextSplitter`)
- Embeds & stores chunks in Pinecone

### 4️⃣ Query answering:

- Searches Pinecone for relevant chunks
- Re-ranks chunks with Pinecone Inference Reranker
- Passes top context to LLM with a **fine-tuned prompt**
- Generates short, direct answers

### 5️⃣ Logging:

- Stores metadata, queries, answers, and response time in MongoDB
- Helps track API activity and logs

# 🔑 Environment Variables

Create a `.env` file or set these in your environment:

`MONGODB_URI=your_mongodb_connection_uri
MONGODB_DB_NAME=your_mongodb_database_name
MONGODB_COLLECTION_NAME=your_mongodb_collection_name
API_AUTHORIZED_TOKEN=your_api_bearer_token

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=us-east-1
INDEX_NAME=your_pinecone_index_name
NAMESPACE=your_pinecone_namespace

TOP_K=10
TOP_N=5

LLM_API_KEY=your_llm_api_key
LLM_MODEL=your_llm_model_name
LLM_BASE_URL=your_llm_base_url

EMBED_MODEL_NAME=your_embedding_model_name
RERANKER_MODEL_NAME=your_reranker_model_name

CHUNK_SIZE=500
CHUNK_OVERLAP=50`
