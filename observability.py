"""
F12: Observability через Langfuse (современный SDK, модуль langfuse.langchain).

Требует переменные окружения в .env:
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com

Получить бесплатный ключ: cloud.langfuse.com -> New Project -> Settings -> API Keys.
Если ключи не заданы, трейсинг просто отключается (граф работает как обычно).
"""

import os
from dotenv import load_dotenv

load_dotenv()

_langfuse_handler = None
_langfuse_enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_handler():
    """
    Возвращает CallbackHandler для передачи в app.invoke(config={"callbacks": [...]}).
    Использует современный модуль langfuse.langchain (работает поверх langchain-core,
    не требует устаревший полный пакет langchain с legacy langchain.callbacks.base).
    Если ключи Langfuse не настроены в .env, возвращает None (трейсинг отключён,
    граф при этом работает штатно).
    """
    global _langfuse_handler

    if not _langfuse_enabled:
        return None

    if _langfuse_handler is None:
        try:
            from langfuse.langchain import CallbackHandler
            _langfuse_handler = CallbackHandler()
        except Exception as e:
            print(f"⚠️ Langfuse недоступен, трейсинг отключён: {e}")
            return None

    return _langfuse_handler


def get_invoke_config(recursion_limit: int = 40) -> dict:
    """
    Собирает config для app.invoke() с трейсингом, если он настроен.
    Использовать так:
        result = app.invoke(initial_state, config=get_invoke_config())
    """
    config = {"recursion_limit": recursion_limit}
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config