from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from agents import extract_text
from llm_utils import safe_invoke

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=3)


def generate_answer(state: AgentState) -> dict:
    prompt = (
        f"Question: {state['question']}\n"
        f"Documents: {state.get('documents')}\n"
        f"SQL: {state.get('sql_result')}\n"
        f"Code execution result: {state.get('code_result')}\n"
        "Write the final answer, grounded ONLY in the evidence above.\n"
        "CRITICAL RULES:\n"
        "- NEVER write, invent, or display Python/SQL code yourself.\n"
        "- Use ONLY the actual 'Code execution result' value if present.\n"
        "- If 'Code execution result' is None/empty and the question needed "
        "computed math, say the calculation could not be completed instead "
        "of fabricating a number or code.\n"
        "- If the evidence does not answer the question, say so explicitly.\n"
        "- If SQL/Code execution result contains a specific number relevant "
        "to the question, you MUST state that exact number in your answer, "
        "not just a qualitative label (e.g. say 'Q4 with revenue 610000', "
        "not just 'Q4')."
    )
    response = safe_invoke(llm, prompt)
    answer = extract_text(response.content)
    return {"answer": answer, "steps": ["generate"]}