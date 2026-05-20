"""
embedding/embedder.py
Tạo embedding model local (HuggingFace — miễn phí, không cần API key).
"""

import torch
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
import config


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Singleton embedder — chỉ load model 1 lần duy nhất.
    Model được tải về local lần đầu, các lần sau dùng cache.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔄 Đang load embedding model: {config.EMBEDDING_MODEL} trên thiết bị: {device} ...")
    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ Embedding model đã sẵn sàng!")
    return embedder
