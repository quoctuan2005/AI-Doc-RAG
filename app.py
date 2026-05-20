"""
app.py — Chainlit UI cho AI Document RAG System
Đây là entry point chính. Chạy bằng: chainlit run app.py
"""

import os
from pathlib import Path
from typing import List

import chainlit as cl

from ingestion.document_loader import load_files
from processing.text_processor import chunk_documents
from vectorstore.faiss_store import build_vectorstore, add_documents
from pipeline.rag_pipeline import build_rag_chain, format_sources
import config


# ─────────────────────────────────────────────────────────────
# HELPER: Xử lý file upload
# ─────────────────────────────────────────────────────────────

async def process_uploaded_files(files) -> int:
    """Xử lý files, build vectorstore, lưu vào session."""
    # Chainlit 2.x: file element có `.path` trỏ tới temp file trên disk
    tmp_paths = [(f.path, f.name) for f in files if f.path]

    # Load documents
    msg = cl.Message(content="📂 **Đang đọc tài liệu...**")
    await msg.send()

    file_paths = [p[0] for p in tmp_paths]
    documents = load_files(file_paths)

    # Đổi metadata source_file về tên gốc
    for tmp_path, original_name in tmp_paths:
        for doc in documents:
            if doc.metadata.get("source_file") == Path(tmp_path).name:
                doc.metadata["source_file"] = original_name

    if not documents:
        await cl.Message(content="❌ Không thể đọc tài liệu. Hãy thử file khác.").send()
        return 0

    # Chunking
    msg.content = f"✂️ **Đang chia nhỏ {len(documents)} trang thành chunks...**"
    await msg.update()
    chunks = chunk_documents(documents)

    # Build / update vector store
    msg.content = f"🧠 **Đang index {len(chunks)} chunks vào vector database...**"
    await msg.update()

    existing_vs = cl.user_session.get("vectorstore")
    if existing_vs is None:
        vectorstore = build_vectorstore(chunks)
    else:
        vectorstore = add_documents(existing_vs, chunks)

    # Build RAG chain
    chain = build_rag_chain(vectorstore)

    # Lưu vào session
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("chain", chain)

    # Chainlit 2.x tự quản lý temp file — không cần xóa thủ công

    return len(chunks)


# ─────────────────────────────────────────────────────────────
# CHAINLIT EVENTS
# ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Khởi động session — hiển thị màn hình chào."""
    cl.user_session.set("vectorstore", None)
    cl.user_session.set("chain", None)

    provider_label = (
        f"Google Gemini (`{config.GOOGLE_MODEL}`)"
        if config.LLM_PROVIDER == "google"
        else f"OpenAI (`{config.OPENAI_MODEL}`)"
    )

    await cl.Message(
        content=f"""# 🤖 AI Document RAG Assistant

Chào mừng bạn! Tôi có thể trả lời câu hỏi dựa trên **tài liệu của bạn**.

---

**⚙️ Cấu hình hiện tại:**
- 🧠 LLM: {provider_label}
- 📐 Embedding: `{config.EMBEDDING_MODEL}`
- 📦 Vector DB: FAISS (local)
- 🔍 Top-K retrieval: `{config.TOP_K}` chunks

---

**📎 Cách sử dụng:**
1. Upload tài liệu (PDF, TXT, DOCX, MD) bằng nút 📎 bên dưới
2. Đặt câu hỏi về nội dung tài liệu
3. Tôi sẽ trả lời có kèm nguồn tham khảo

> Bạn có thể upload nhiều file và hỏi nhiều câu liên tiếp — tôi nhớ lịch sử hội thoại!
""",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Xử lý mỗi tin nhắn từ người dùng."""

    # ── Xử lý file upload ────────────────────────────────────
    if message.elements:
        files = [
            el for el in message.elements
            if hasattr(el, "path") and el.path and Path(el.name).suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]

        if files:
            file_names = ", ".join(f"`{f.name}`" for f in files)
            await cl.Message(content=f"📎 Nhận được: {file_names}").send()

            chunk_count = await process_uploaded_files(files)

            if chunk_count > 0:
                file_list = "\n".join(f"  • {f.name}" for f in files)
                await cl.Message(
                    content=f"""✅ **Xử lý hoàn tất!**

**Files đã index:**
{file_list}

📊 Tổng cộng **{chunk_count} chunks** đã được lưu vào vector database.

💬 Hãy đặt câu hỏi về nội dung tài liệu!"""
                ).send()
            return

    # ── Kiểm tra chain đã được khởi tạo chưa ───────────────
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(
            content="⚠️ **Chưa có tài liệu nào được upload.**\n\nHãy upload file PDF/TXT/DOCX trước bằng nút 📎 bên dưới nhé!"
        ).send()
        return

    # ── Chạy RAG pipeline ────────────────────────────────────
    thinking_msg = cl.Message(content="🔍 Đang tìm kiếm trong tài liệu...")
    await thinking_msg.send()

    try:
        # RAGChain.invoke là synchronous — dùng make_async để không block event loop
        result = await cl.make_async(chain.invoke)({"question": message.content})

        answer = result.get("answer", "Không tìm thấy câu trả lời.")
        source_docs = result.get("source_documents", [])
        sources_text = format_sources(source_docs)

        final_content = answer
        if sources_text:
            final_content += f"\n\n{sources_text}"

        thinking_msg.content = final_content
        await thinking_msg.update()

    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"RAG Error:\n{err_detail}")
        thinking_msg.content = f"❌ Có lỗi xảy ra: `{str(e)}`\n\nHãy kiểm tra lại API key trong file `.env`."
        await thinking_msg.update()
