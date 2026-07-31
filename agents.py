import os
import io
import contextlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import TavilySearchResults
from ingest import get_vectorstore
from state import AgentState
from func_timeout import func_timeout, FunctionTimedOut
from llm_utils import safe_invoke

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=3,
)

# Абсолютный путь к БД: работает независимо от того, откуда запущен скрипт
# (main.py из корня проекта ИЛИ eval/evaluate.py из подпапки eval/).
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "company.db"


def extract_text(content) -> str:
    """Приводит content ответа LLM (строка ИЛИ список блоков) к чистой строке."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def run_python_sandboxed(code: str) -> str:
    """
    Выполняет Python-код в едином namespace (без раздельных globals/locals),
    чтобы избежать NameError внутри функций (известная проблема PythonREPL).
    Возвращает захваченный stdout.
    """
    namespace = {"__builtins__": __builtins__}
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        exec(code, namespace)
    return stdout_capture.getvalue()


# --- F3: Retriever Agent ---
def retriever_agent(state: AgentState) -> dict:
    """Извлекает релевантные чанки из Qdrant. Не роняет граф, если коллекции нет."""
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(state["question"])
        doc_contents = [d.page_content for d in docs]
        return {
            "documents": state.get("documents", []) + doc_contents,
            "steps": ["retriever"],
        }
    except Exception as e:
        return {
            "documents": state.get("documents", []) + [f"Retriever unavailable: {e}"],
            "steps": ["retriever_error"],
        }


# --- F4: Web Agent ---
def web_agent(state: AgentState) -> dict:
    """Выполняет поиск в веб через Tavily."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return {
            "documents": state.get("documents", []) + ["Web search skipped: TAVILY_API_KEY is missing."],
            "steps": ["web_skipped"],
        }
    try:
        search = TavilySearchResults(k=3)
        results = search.invoke({"query": state["question"]})
        web_docs = [r["content"] for r in results]
        return {
            "documents": state.get("documents", []) + web_docs,
            "steps": ["web"],
        }
    except Exception as e:
        return {
            "documents": state.get("documents", []) + [f"Web search error: {e}"],
            "steps": ["web_error"],
        }


# --- F5: Data Agent (Text-to-SQL) ---
def data_agent(state: AgentState) -> dict:
    """Пишет и выполняет ТОЛЬКО SELECT запросы к SQLite."""
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.as_posix()}")

    prompt = f"""You are a SQL expert.
Write a single read-only SQLite query to answer the question based on schema:
{db.get_table_info()}

Question: {state['question']}
Relevant past conversation context (use it to resolve references like
"previous quarter", "that customer", "and before that"): {state.get('memory_context', '')}

IMPORTANT: If the question requires further calculation (std deviation, average,
custom math), return the RAW rows needed (e.g. SELECT quarter, revenue FROM metrics),
do NOT try to compute statistics like STDDEV inside SQL - SQLite does not support
STDDEV natively and it will silently return wrong results.

IMPORTANT: If the question asks to identify a row by an extreme value (e.g.
"which quarter had the highest/lowest revenue"), you MUST include the compared
numeric column itself in the SELECT (e.g. SELECT quarter, revenue FROM metrics
ORDER BY revenue DESC LIMIT 1), not just the label column. The final answer
needs that number as evidence, not only the winning row's name.

Return ONLY raw SQL statement. No markdown code blocks, no explanations."""

    raw = safe_invoke(llm, prompt).content
    sql_query = extract_text(raw).strip()
    clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()

    if not clean_sql.lower().startswith("select"):
        return {
            "sql_result": "Error: Only SELECT operations are allowed.",
            "steps": ["data(sql_blocked)"],
        }

    try:
        result = db.run(clean_sql)
        return {
            "sql_result": f"SQL: {clean_sql}\nResult: {result}",
            "steps": ["data(sql)"],
        }
    except Exception as e:
        return {
            "sql_result": f"SQL Error: {str(e)}",
            "steps": ["data(sql_error)"],
        }


# --- F6: Code Agent (собственный sandbox, cap 10 сек) ---
def code_agent(state: AgentState) -> dict:
    """Выполняет Python-код для вычислений, с капом времени выполнения."""
    prompt = f"""Write executable Python code to answer the question.
Question: {state['question']}
SQL Result Context: {state.get('sql_result', 'None')}

Requirements:
1. Use ONLY the Python standard library (e.g. statistics, math, json, re).
   Do NOT import pandas, numpy, or any other third-party package.
2. Do NOT define functions - write plain top-level statements only.
3. Parse any numbers you need directly out of the SQL Result Context text above.
4. Print the final answer using print().
5. Return ONLY clean Python code without markdown tags."""

    raw = safe_invoke(llm, prompt).content
    code = extract_text(raw).strip()
    clean_code = code.replace("```python", "").replace("```", "").strip()

    try:
        exec_res = func_timeout(10, run_python_sandboxed, args=(clean_code,))
        return {"code_result": exec_res.strip(), "steps": ["code"]}
    except FunctionTimedOut:
        return {"code_result": "Error: code execution timed out (10s cap)", "steps": ["code_timeout"]}
    except Exception as e:
        return {"code_result": f"Python Exec Error: {e}", "steps": ["code_error"]}