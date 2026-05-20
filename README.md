1️⃣ Kiến trúc tổng thể của hệ thống RAG
                    ┌──────────────────┐
                    │   User / Client  │
                    │ Web / API / App  │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │     API Layer    │
                    │ FastAPI / Flask  │
                    └─────────┬────────┘
                              │
                     Query Processing
                              │
                              ▼
               ┌────────────────────────┐
               │     RAG Orchestrator   │
               │  (LangChain / Custom)  │
               └─────────┬──────────────┘
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
 Query Embedding   Vector Retrieval     Prompt Builder
       │                 │                 │
       ▼                 ▼                 ▼
Embedding Model     Vector Database      Context
       │             (FAISS/Chroma)       │
       └───────────────┬──────────────────┘
                       ▼
                Large Language Model
              (GPT / Llama / Mistral)
                       │
                       ▼
                   Final Answer
                       │
                       ▼
                    User
2️⃣ Kiến trúc chi tiết (5 Layer chính)

Một hệ thống RAG production thường chia thành 5 tầng.

1. Data Ingestion Layer
2. Data Processing Layer
3. Vector Storage Layer
4. Retrieval & Reasoning Layer
5. Application Layer
3️⃣ Layer 1 — Data Ingestion (Document ingestion)

Nhiệm vụ:

thu thập tài liệu
convert sang text

Nguồn dữ liệu:

PDF
Word
HTML
Database
CSV
Notion
Confluence
GitHub

Pipeline:

Document → Parser → Raw Text

Tools phổ biến:

PyMuPDF
Unstructured
Apache Tika
LangChain loaders

Ví dụ module:

ingestion/
    pdf_loader.py
    web_scraper.py
    docx_loader.py
4️⃣ Layer 2 — Document Processing

Sau khi load tài liệu cần:

1️⃣ Cleaning
remove headers
remove page numbers
remove noise
2️⃣ Chunking

Chia document thành đoạn nhỏ.

Ví dụ:

chunk_size = 500
overlap = 50

Pipeline:

Document
   ↓
Chunking
   ↓
Chunks
3️⃣ Metadata enrichment

Thêm metadata:

source
page_number
title
author
timestamp

Ví dụ:

{
 "text": "...",
 "source": "employee_handbook.pdf",
 "page": 15
}
5️⃣ Layer 3 — Embedding & Vector Storage
Embedding model

Chuyển text → vector

Popular models:

OpenAI text-embedding
bge-large
e5-large
sentence-transformers

Pipeline:

Chunk → Embedding → Vector
Vector Database

Lưu vector để search nhanh.

Các DB phổ biến:

Vector DB	Use case
FAISS	local project
Chroma	lightweight
Pinecone	cloud
Weaviate	production
Milvus	large scale

Vector structure:

{
 vector: [0.213,0.845,...],
 text: "policy content",
 metadata: {page:12}
}
6️⃣ Layer 4 — Retrieval Pipeline

Khi user hỏi:

User Query

Pipeline:

Query
 ↓
Query Embedding
 ↓
Vector Similarity Search
 ↓
Top K Chunks
 ↓
Context Builder
 ↓
Prompt Construction
 ↓
LLM
 ↓
Answer
Retrieval Techniques

Basic:

Vector Similarity

Advanced:

Hybrid Search (BM25 + vector)
Reranking
Multi-query retrieval
Query rewriting
Reranking Layer (advanced)

Sau khi search:

Top 20 documents
↓
Reranker model
↓
Top 5 best context

Models:

bge-reranker
cohere rerank
cross-encoder
7️⃣ Layer 5 — LLM Generation

LLM nhận:

Question
+
Retrieved Context

Prompt:

Answer the question using the provided context.
If the answer is not in the context, say you don't know.

Ví dụ:

Context:
{docs}

Question:
{query}

LLM:

GPT4
Llama3
Mistral
Claude
8️⃣ Application Layer

Đây là phần người dùng tương tác.

Ví dụ:

Web Chatbot
Slack Bot
API
Mobile App

Stack phổ biến:

FastAPI
Streamlit
NextJS
React
9️⃣ Kiến trúc microservices (production)

Project lớn thường tách service.

                    User
                     │
                     ▼
               API Gateway
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
Document Service   Query Service   Auth Service
     │               │
     ▼               ▼
Processing        Retrieval
     │               │
     ▼               ▼
Embedding Service   Vector DB
                     │
                     ▼
                     LLM
🔟 Project structure chuẩn
rag-system
│
├── ingestion
│   ├── pdf_loader.py
│   ├── web_loader.py
│
├── processing
│   ├── chunking.py
│   ├── cleaning.py
│
├── embedding
│   └── embedding_model.py
│
├── vectorstore
│   └── vector_db.py
│
├── retrieval
│   ├── retriever.py
│   ├── reranker.py
│
├── llm
│   └── llm_client.py
│
├── pipeline
│   └── rag_pipeline.py
│
├── api
│   └── app.py
│
└── ui
    └── streamlit_app.py
1️⃣1️⃣ Pipeline end-to-end
Documents
   ↓
Parsing
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Vector DB
   ↓
User Query
   ↓
Embedding Query
   ↓
Vector Search
   ↓
Reranking
   ↓
Prompt
   ↓
LLM
   ↓
Answer
1️⃣2️⃣ Kiến trúc nâng cao (RAG hiện đại)

Các hệ thống tiên tiến như Perplexity hoặc OpenAI sử dụng:

Advanced RAG

Thêm:

Query rewriting
Multi retrieval
Reranking
Tool calling
Memory
Caching
Evaluation