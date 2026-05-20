"""
ingestion/document_loader.py
Thu thập và load tài liệu từ nhiều định dạng khác nhau.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

import config


def load_file(file_path: str) -> List[Document]:
    """Load một file đơn lẻ, tự động nhận diện định dạng."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Định dạng không được hỗ trợ: {ext}")

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in (".txt",):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        except ImportError:
            raise ImportError("Cần cài: pip install docx2txt")
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()

    # Thêm metadata: tên file
    for doc in docs:
        doc.metadata["source_file"] = path.name

    return docs


def load_files(file_paths: List[str]) -> List[Document]:
    """Load nhiều file, trả về danh sách Document gộp lại."""
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            docs = load_file(fp)
            all_docs.extend(docs)
            print(f"✅ Loaded {len(docs)} pages từ: {Path(fp).name}")
        except Exception as e:
            print(f"❌ Lỗi khi load {fp}: {e}")
    return all_docs
