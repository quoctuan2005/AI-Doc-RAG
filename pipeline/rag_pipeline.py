"""
pipeline/rag_pipeline.py
Kết hợp tất cả layers thành RAG pipeline hoàn chỉnh.
Hỗ trợ: conversational retrieval (nhớ lịch sử hội thoại).
"""

from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from llm.llm_client import get_llm
import config

# ── System Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là một AI Assistant thông minh chuyên phân tích tài liệu.

Quy tắc:
- Chỉ trả lời dựa trên context tài liệu được cung cấp.
- Nếu context không đủ thông tin, hãy nói rõ là bạn không tìm thấy thông tin đó trong tài liệu.
- Trả lời ngắn gọn, rõ ràng, có cấu trúc (dùng bullet points nếu cần).
- Nếu câu hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt.

Context từ tài liệu:
{context}"""

CONDENSE_QUESTION_PROMPT = """Dựa vào lịch sử hội thoại bên dưới và câu hỏi mới nhất, hãy tạo ra một câu hỏi độc lập (standalone question) bằng ngôn ngữ gốc của câu hỏi.
Câu hỏi độc lập này phải tự chứa đầy đủ ngữ cảnh để tìm kiếm thông tin mà không cần xem lại lịch sử hội thoại.

Lịch sử hội thoại:
{chat_history}

Câu hỏi mới nhất: {question}
Câu hỏi độc lập:"""


class RAGChain:
    """
    RAG Chain tự quản lý lịch sử hội thoại.
    Dùng LCEL (LangChain Expression Language) hiện đại.
    """

    def __init__(self, vectorstore: FAISS):
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K},
        )
        self.llm = get_llm()
        self.chat_history: List[Dict] = []  # [{"role": "human"|"ai", "content": str}]

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _build_messages(self, question: str, context: str) -> List:
        messages = [
            ("system", SYSTEM_PROMPT.format(context=context)),
        ]
        # Thêm lịch sử hội thoại (5 lượt gần nhất)
        for turn in self.chat_history[-5:]:
            messages.append((turn["role"], turn["content"]))
        messages.append(("human", question))
        return messages

    def invoke(self, inputs: Dict) -> Dict:
        question = inputs["question"]

        # Viết lại câu hỏi nếu có lịch sử hội thoại
        query_for_retrieval = question
        if self.chat_history:
            history_str = "\n".join(
                f"{'Human' if turn['role'] == 'human' else 'Assistant'}: {turn['content']}"
                for turn in self.chat_history[-5:]
            )
            condense_prompt = CONDENSE_QUESTION_PROMPT.format(
                chat_history=history_str, question=question
            )
            try:
                condensed = self.llm.invoke(condense_prompt)
                query_for_retrieval = condensed.content.strip()
                print(f"🔄 Viết lại câu hỏi: '{question}' -> '{query_for_retrieval}'")
            except Exception as e:
                print(f"⚠️ Lỗi khi viết lại câu hỏi: {e}")

        # Retrieve relevant docs
        docs = self.retriever.invoke(query_for_retrieval)
        context = self._format_docs(docs)

        # Build messages với history
        messages = self._build_messages(question, context)

        # Call LLM
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        lc_messages = []
        for role, content in messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role == "ai":
                lc_messages.append(AIMessage(content=content))

        response = self.llm.invoke(lc_messages)
        answer = response.content

        # Lưu vào history
        self.chat_history.append({"role": "human", "content": question})
        self.chat_history.append({"role": "ai", "content": answer})

        return {"answer": answer, "source_documents": docs}


def build_rag_chain(vectorstore: FAISS) -> RAGChain:
    """Tạo RAGChain từ vectorstore."""
    return RAGChain(vectorstore)


def format_sources(source_docs: List[Document]) -> str:
    """Format danh sách source documents thành text hiển thị."""
    if not source_docs:
        return ""

    seen = set()
    lines = ["---", "📚 **Nguồn tham khảo:**"]

    for doc in source_docs:
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f" — trang {page + 1}" if page != "" else ""
        key = f"{source_file}{page_str}"

        if key not in seen:
            seen.add(key)
            lines.append(f"• `{source_file}`{page_str}")

    return "\n".join(lines)
