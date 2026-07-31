from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from llm_utils import safe_invoke
from memory import recall_relevant

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=3)

MAX_AGENT_CALLS = 6
CODE_KEYWORDS = ["python", "код", "выполни код", "напиши код", "посчитай", "вычисли", "std", "отклонение"]
DATA_KEYWORDS = [
    "сколько", "какой", "какая", "какие", "средн", "revenue", "выручк",
    "клиент", "customer", "churn", "отток", "квартал", "quarter",
]
RETRIEVER_KEYWORDS = [
    "расскажи", "компани", "продукт", "поддержк", "support", "возврат",
    "refund", "email", "почта", "политик", "policy",
]


class Route(BaseModel):
    next: str = Field(description="one of: retriever, web, data, code, finish")


def supervisor(state: AgentState) -> dict:
    steps = state.get("steps", [])
    question_lower = state["question"].lower()
    agent_calls = sum(1 for s in steps if s in ("retriever", "web", "data(sql)", "code"))

    if not steps:
        past_turns = recall_relevant(state["question"], k=3)
        memory_context = "\n".join(past_turns) if past_turns else ""
    else:
        memory_context = state.get("memory_context") or ""

    if agent_calls >= MAX_AGENT_CALLS:
        return {"plan": "finish", "memory_context": memory_context, "steps": ["supervisor→finish(forced)"]}

    last_step = steps[-1] if steps else None
    if last_step in ("code_error", "code_timeout"):
        return {"plan": "finish", "memory_context": memory_context, "steps": ["supervisor→finish(code_failed)"]}

    has_sql = any(s == "data(sql)" for s in steps)
    has_code = any(s == "code" for s in steps)
    has_retrieved = any(s == "retriever" for s in steps)
    wants_code = any(kw in question_lower for kw in CODE_KEYWORDS)
    wants_data = any(kw in question_lower for kw in DATA_KEYWORDS)
    wants_retriever = any(kw in question_lower for kw in RETRIEVER_KEYWORDS)

    # Детерминированные правила приоритетнее решения слабой LLM.
    # Порядок проверки важен: code только после data; data и retriever
    # проверяются на "чистом" первом шаге, чтобы не конфликтовать.
    if wants_code and has_sql and not has_code:
        return {"plan": "code", "memory_context": memory_context, "steps": ["supervisor→code(forced)"]}

    if not steps:
        # RETRIEVER_KEYWORDS содержательные и специфичные (компания/продукт/
        # поддержка/возврат/политика) — если хоть одно совпало, это сильнее
        # сигнала, чем generic-слова вроде "какой/какая" из DATA_KEYWORDS,
        # которые есть почти в любом вопросе на русском.
        if wants_retriever:
            return {"plan": "retriever", "memory_context": memory_context, "steps": ["supervisor→retriever(forced)"]}
        if wants_data:
            return {"plan": "data", "memory_context": memory_context, "steps": ["supervisor→data(forced)"]}

    prompt = (
        f"Question: {state['question']}\n"
        f"Relevant past turns (long-term memory): {memory_context}\n"
        f"Steps taken so far: {steps}\n"
        f"Collected documents: {bool(state.get('documents'))}\n"
        f"SQL result so far: {state.get('sql_result')}\n"
        f"Code result so far: {state.get('code_result')}\n"
        f"Critic feedback (if revision requested): {state.get('critic_reason')}\n"
        "Decide the single next agent to run, or 'finish' if enough evidence "
        "has been gathered to answer the question. "
        "Use past turns only if the current question refers to earlier context "
        "(e.g. 'a предыдущий квартал?', 'and before that?').\n"
        "IMPORTANT: do not call the same agent twice in a row unless it failed. "
        "If the question needs numbers from the database, call 'data' first. "
        "If the question is about the company's info/product/support/policy, call 'retriever' first."
    )
    structured_llm = llm.with_structured_output(Route)
    decision = safe_invoke(structured_llm, prompt)
    next_agent = decision.next

    if next_agent == "code" and not has_sql:
        next_agent = "data"

    return {"plan": next_agent, "memory_context": memory_context, "steps": [f"supervisor→{next_agent}"]}