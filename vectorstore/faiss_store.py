"""
vectorstore/faiss_store.py
Quản lý FAISS vector database — lưu và tìm kiếm vector.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from embedding.embedder import get_embedder


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Tạo FAISS vectorstore từ danh sách chunks."""
    embedder = get_embedder()
    print(f"⚙️  Đang index {len(chunks)} chunks vào FAISS...")
    vectorstore = FAISS.from_documents(chunks, embedder)
    print("✅ Vector store đã sẵn sàng!")
    return vectorstore


def add_documents(vectorstore: FAISS, chunks: List[Document]) -> FAISS:
    """Thêm documents mới vào vectorstore hiện có."""
    embedder = get_embedder()
    vectorstore.add_documents(chunks)
    return vectorstore


def similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """Tìm kiếm top-k chunks tương đồng nhất với query."""
    return vectorstore.similarity_search(query, k=k)
