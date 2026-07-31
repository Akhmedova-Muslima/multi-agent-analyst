"""
eval/compare_critic.py

Обязательный визуал по рубрике: "the evaluation metrics table (with vs
without the critic)".

Строит ВТОРОЙ граф - идентичный основному, но с critic вырезанным:
generate -> END напрямую, без верификации и без revision-петли.
Прогоняет тот же testset.json через оба графа и печатает сравнительную
таблицу метрик.

Запуск:
  cd eval
  python compare_critic.py
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from state import AgentState
from agents import retriever_agent, web_agent, data_agent, code_agent
from supervisor import supervisor
from generate import generate_answer
from graph import app as app_with_critic

from evaluate import judge_answer, TESTSET_PATH, PAUSE_BETWEEN_QUESTIONS

RESULTS_PATH = Path(__file__).parent / "eval_compare_results.json"


# ---------- граф БЕЗ critic ----------
_g = StateGraph(AgentState)
_g.add_node("supervisor", supervisor)
_g.add_node("retriever", retriever_agent)
_g.add_node("web", web_agent)
_g.add_node("data", data_agent)
_g.add_node("code", code_agent)
_g.add_node("generate", generate_answer)

_g.set_entry_point("supervisor")
_g.add_conditional_edges(
    "supervisor",
    lambda s: s["plan"],
    {
        "retriever": "retriever",
        "web": "web",
        "data": "data",
        "code": "code",
        "finish": "generate",
    },
)
for _a in ["retriever", "web", "data", "code"]:
    _g.add_edge(_a, "supervisor")
_g.add_edge("generate", END)  # <-- ключевое отличие: без critic

app_no_critic = _g.compile()


def run_single_question(graph_app, question: str) -> dict:
    initial_state = {
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
    try:
        result = graph_app.invoke(initial_state, config={"recursion_limit": 40})
        return {
            "answer": result.get("answer", ""),
            "steps": result.get("steps", []),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "steps": [], "error": str(e)}


class _FallbackScore:
    correctness = 0
    groundedness = 0
    relevance = 0
    comment = "judge failed"


def evaluate_variant(graph_app, testset, label: str) -> list:
    results = []
    print(f"\n=== Вариант: {label} ===\n")
    for i, item in enumerate(testset, start=1):
        question = item["question"]
        reference = item["reference_answer"]
        print(f"[{label} {i}/{len(testset)}] {question}")

        run_result = run_single_question(graph_app, question)
        if run_result["error"]:
            print(f"   ❌ Ошибка графа: {run_result['error']}")
            results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "category": item.get("category", "unknown"),
                    "answer": "",
                    "correctness": 0,
                    "groundedness": 0,
                    "relevance": 0,
                }
            )
            time.sleep(PAUSE_BETWEEN_QUESTIONS)
            continue

        actual_answer = run_result["answer"]
        print(f"   Ответ: {actual_answer[:120]}")

        try:
            score = judge_answer(question, reference, actual_answer)
            print(
                f"   correctness={score.correctness} "
                f"groundedness={score.groundedness} relevance={score.relevance}"
            )
        except Exception as e:
            print(f"   ⚠️ Judge failed: {e}")
            score = _FallbackScore()

        results.append(
            {
                "id": item["id"],
                "question": question,
                "category": item.get("category", "unknown"),
                "answer": actual_answer,
                "correctness": score.correctness,
                "groundedness": score.groundedness,
                "relevance": score.relevance,
            }
        )
        time.sleep(PAUSE_BETWEEN_QUESTIONS)

    return results


def summarize(results, label: str):
    n = len(results)
    avg_c = sum(r["correctness"] for r in results) / n
    avg_g = sum(r["groundedness"] for r in results) / n
    avg_r = sum(r["relevance"] for r in results) / n
    print(f"\n{label}: correctness={avg_c:.2f}  groundedness={avg_g:.2f}  relevance={avg_r:.2f}")
    return {"correctness": avg_c, "groundedness": avg_g, "relevance": avg_r}


def main():
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))

    with_critic_results = evaluate_variant(app_with_critic, testset, "С CRITIC")

    print("\n⏸️  Пауза 90с между вариантами — даём квоте восстановиться...\n")
    time.sleep(90)

    without_critic_results = evaluate_variant(app_no_critic, testset, "БЕЗ CRITIC")

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: С CRITIC vs БЕЗ CRITIC")
    print("=" * 70)
    wc = summarize(with_critic_results, "С critic   ")
    woc = summarize(without_critic_results, "Без critic  ")

    print("\n" + f"{'Метрика':<15}{'С critic':<12}{'Без critic':<12}{'Δ':<8}")
    for metric in ("correctness", "groundedness", "relevance"):
        delta = wc[metric] - woc[metric]
        print(f"{metric:<15}{wc[metric]:<12.2f}{woc[metric]:<12.2f}{delta:+.2f}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "with_critic": with_critic_results,
                "without_critic": without_critic_results,
                "summary": {"with_critic": wc, "without_critic": woc},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n✅ Полные результаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    main()