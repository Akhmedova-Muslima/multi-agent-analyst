import time


def safe_invoke(llm, prompt, max_attempts: int = 5, base_wait: int = 20):
    """
    Вызывает llm.invoke(prompt) с ручным retry.
    Ловит и квоту (429/RESOURCE_EXHAUSTED), и временные сетевые сбои
    (Server disconnected, timeout, connection reset и т.п.) —
    всё это транзиентные ошибки на бесплатном тарифе Gemini.
    Не ретраит только явно "непоправимые" ошибки (404 модели, авторизация).
    """
    NON_RETRYABLE_MARKERS = ["NOT_FOUND", "404", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403"]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            msg = str(e)

            if any(marker in msg for marker in NON_RETRYABLE_MARKERS):
                raise

            wait = base_wait * attempt
            print(f"⏳ Ошибка при вызове LLM (попытка {attempt}/{max_attempts}): {type(e).__name__}. Жду {wait}с...")
            time.sleep(wait)

    raise last_error