from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from llm_utils import safe_invoke

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=3)


class Verdict(BaseModel):
    ok: bool
    reason: str


def critic(state: AgentState) -> dict:
    prompt = (
        f"Question: {state['question']}\n"
        f"Evidence — documents: {state.get('documents')}\n"
        f"SQL: {state.get('sql_result')}\n"
        f"Code: {state.get('code_result')}\n"
        f"Draft answer: {state.get('answer')}\n"
        "Is this answer correct AND fully supported by the evidence above? "
        "ok=true only if fully grounded, otherwise ok=false."
    )
    structured_llm = llm.with_structured_output(Verdict)
    verdict = safe_invoke(structured_llm, prompt)
    return {
        "approved": verdict.ok,
        "critic_reason": verdict.reason,
        "revisions": state.get("revisions", 0) + (0 if verdict.ok else 1),
        "steps": [f"critic:{'ok' if verdict.ok else 'revise: ' + verdict.reason}"],
    }