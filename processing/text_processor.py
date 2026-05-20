"""
processing/text_processor.py
Làm sạch text và chia thành chunks.
"""

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def clean_text(text: str) -> str:
    """Loại bỏ noise: khoảng trắng thừa, ký tự lạ."""
    # Xóa nhiều dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Xóa khoảng trắng thừa đầu/cuối mỗi dòng
    text = "\n".join(line.strip() for line in text.splitlines())
    # Xóa ký tự không in được
    text = re.sub(r"[^\x20-\x7E\n\u00C0-\u024F\u0100-\u017F\u1EA0-\u1EF9]", " ", text)
    return text.strip()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chia documents thành chunks nhỏ.
    Sử dụng RecursiveCharacterTextSplitter để giữ ngữ nghĩa tốt nhất.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )

    chunks: List[Document] = []
    for doc in documents:
        # Clean text trước khi chunk
        doc.page_content = clean_text(doc.page_content)

        if not doc.page_content.strip():
            continue

        split_docs = splitter.split_documents([doc])

        # Enrich metadata: thêm chunk index
        for i, chunk in enumerate(split_docs):
            chunk.metadata["chunk_index"] = i
            chunk.metadata.setdefault("source_file", "unknown")

        chunks.extend(split_docs)

    print(f"📄 Tổng chunks: {len(chunks)}")
    return chunks
