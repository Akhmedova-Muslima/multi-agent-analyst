"""
api_server.py
--------------
FastAPI-обёртка над LangGraph-приложением (F13: streaming frontend).

Соответствует реальному state.py:
  question, plan, documents, sql_result, code_result, answer,
  approved, critic_reason, memory_context,
  steps: Annotated[List[str], operator.add]   <-- LangGraph сам суммирует
  revisions: int

Ключевая деталь: раз steps суммируется через operator.add на уровне графа,
самый надёжный способ получить ПОЛНОЕ накопленное состояние после каждого
шага — это stream_mode="values" (не "updates"), которое отдаёт полный
снимок state после каждого супер-шага. Дальше мы просто сравниваем длину
steps с предыдущим снимком, чтобы понять, что нового появилось, и шлём
это в SSE.

Запуск:
  uvicorn api_server:app --reload --port 8000
"""

import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import app as agent_graph
from state import AgentState
from memory import save_turn
from observability import get_invoke_config

app = FastAPI(title="Multi-Agent Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # на проде сузь до домена фронта (Vercel URL)
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_graph(question: str, session_id: str) -> AsyncGenerator[str, None]:
    initial_state: AgentState = {
        "question": question,
        "plan": None,
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": None,
        "approved": None,
        "critic_reason": None,
        "memory_context": None,
        "steps": [],
        "revisions": 0,
    }

    config = get_invoke_config(recursion_limit=40)
    # если get_invoke_config поддерживает session/trace id — можно прокинуть его сюда,
    # чтобы SSE-сессии тоже трейсились в Langfuse отдельными записями:
    # config = get_invoke_config(recursion_limit=40, session_id=session_id)

    seen_steps = 0
    final_state: dict = initial_state

    try:
        async for state in agent_graph.astream(
            initial_state, config=config, stream_mode="values"
        ):
            final_state = state
            steps = state.get("steps", [])

            # шлём только НОВЫЕ шаги, появившиеся с прошлого снимка
            new_steps = steps[seen_steps:]
            seen_steps = len(steps)

            for step_label in new_steps:
                yield sse_event(
                    "step",
                    {
                        "label": step_label,
                        "steps_so_far": steps,
                        "sql_result": state.get("sql_result"),
                        "code_result": state.get("code_result"),
                        "plan": state.get("plan"),
                        "approved": state.get("approved"),
                        "critic_reason": state.get("critic_reason"),
                        "revisions": state.get("revisions", 0),
                    },
                )

        answer = final_state.get("answer") or ""
        # main.py делает то же самое после app.invoke — сохраняем ход в память
        save_turn(question, answer)

        yield sse_event(
            "final",
            {
                "answer": answer,
                "steps": final_state.get("steps", []),
                "revisions": final_state.get("revisions", 0),
                "approved": final_state.get("approved"),
            },
        )

    except Exception as e:
        yield sse_event("error", {"message": str(e)})


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        stream_graph(req.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}