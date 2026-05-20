"""
llm/llm_client.py
Khởi tạo LLM client dựa trên config: Google Gemini hoặc OpenAI.
"""

from langchain_core.language_models.chat_models import BaseChatModel
import config


def get_llm() -> BaseChatModel:
    """
    Trả về LLM phù hợp dựa trên LLM_PROVIDER trong .env.
    Hỗ trợ: 'google' (Gemini) | 'openai' (GPT)
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "google":
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY chưa được set trong .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY chưa được set trong .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )

    else:
        raise ValueError(f"LLM_PROVIDER không hợp lệ: '{provider}'. Dùng 'google' hoặc 'openai'.")
