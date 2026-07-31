from langgraph.graph import StateGraph, END
from state import AgentState
from agents import retriever_agent, web_agent, data_agent, code_agent
from supervisor import supervisor
from critic import critic
from generate import generate_answer

MAX_REVISIONS = 3

def route_after_critic(state: AgentState) -> str:
    if state.get("approved"):
        return "finish"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "finish"
    return "revise"

g = StateGraph(AgentState)
g.add_node("supervisor", supervisor)
g.add_node("retriever", retriever_agent)
g.add_node("web", web_agent)
g.add_node("data", data_agent)
g.add_node("code", code_agent)
g.add_node("generate", generate_answer)
g.add_node("critic", critic)

g.set_entry_point("supervisor")
g.add_conditional_edges("supervisor", lambda s: s["plan"], {
    "retriever": "retriever",
    "web": "web",
    "data": "data",
    "code": "code",
    "finish": "generate",
})
for a in ["retriever", "web", "data", "code"]:
    g.add_edge(a, "supervisor")
g.add_edge("generate", "critic")
g.add_conditional_edges("critic", route_after_critic, {"finish": END, "revise": "supervisor"})

app = g.compile()
