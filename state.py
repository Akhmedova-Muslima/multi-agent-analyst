from typing import TypedDict, List, Optional, Annotated
import operator

class AgentState(TypedDict):
    question: str
    plan: Optional[str]
    documents: List[str]
    sql_result: Optional[str]
    code_result: Optional[str]
    answer: Optional[str]
    approved: Optional[bool]
    critic_reason: Optional[str]
    memory_context: Optional[str]
    steps: Annotated[List[str], operator.add]
    revisions: int