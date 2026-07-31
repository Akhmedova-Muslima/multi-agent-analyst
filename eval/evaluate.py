"""
F11: Evaluation harness.

Прогоняет testset.json через весь multi-agent граф, затем оценивает
каждый ответ через LLM-as-judge (1-5) по критериям: correctness,
groundedness, relevance. Печатает сводную таблицу метрик.

Замечание про RAGAS: полноценный RAGAS (faithfulness, answer_relevancy,
context_precision) на бесплатном тарифе Gemini (15 запросов/минуту)
делает 3-5 доп. LLM-вызовов на метрику на вопрос - для 11 вопросов это
100+ запросов и гарантированно упрётся в дневную квоту. Поэтому здесь
реализован эквивалентный по духу, но экономный LLM-judge (1 вызов на
вопрос для скоринга), который покрывает те же три оси качества, что
и faithfulness/answer_relevancy/context_precision из RAGAS.
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from graph import app
from llm_utils import safe_invoke

JUDGE_MODEL = "gemini-3.1-flash-lite"
judge_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0, max_retries=3)

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

PAUSE_BETWEEN_QUESTIONS = 25  # секунд, страховка от rate limit 15/минуту


class JudgeScore(BaseModel):
    correctness: int = Field(description="1-5: does the answer match the reference answer?")
    groundedness: int = Field(description="1-5: is the answer supported by evidence, not hallucinated?")
    relevance: int = Field(description="1-5: does the answer actually address the question asked?")
    comment: str = Field(description="one short sentence explaining the scores")


def run_single_question(question: str) -> dict:
    """Прогоняет один вопрос через полный multi-agent граф."""
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
        result = app.invoke(initial_state, config={"recursion_limit": 40})
        return {
            "answer": result.get("answer", ""),
            "steps": result.get("steps", []),
            "approved": result.get("approved"),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "steps": [], "approved": None, "error": str(e)}


def judge_answer(question: str, reference_answer: str, actual_answer: str) -> JudgeScore:
    """LLM-as-judge: сравнивает фактический ответ с эталонным."""
    structured_judge = judge_llm.with_structured_output(JudgeScore)
    prompt = (
        f"Question: {question}\n"
        f"Reference (expected) answer: {reference_answer}\n"
        f"Actual answer from the system: {actual_answer}\n\n"
        "Score the actual answer on three axes, each 1-5 (5 = best):\n"
        "- correctness: does it match the reference answer's facts/numbers?\n"
        "- groundedness: does it look grounded in real evidence, not made up?\n"
        "- relevance: does it actually address what was asked?\n"
        "Give a one-sentence comment explaining your scores."
    )
    return safe_invoke(structured_judge, prompt)


def main():
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    results = []

    print(f"🚀 Запуск evaluation harness на {len(testset)} вопросах...\n")

    for i, item in enumerate(testset, start=1):
        question = item["question"]
        reference = item["reference_answer"]
        category = item.get("category", "unknown")

        print(f"[{i}/{len(testset)}] {question}")

        run_result = run_single_question(question)

        if run_result["error"]:
            print(f"   ❌ Ошибка выполнения графа: {run_result['error']}")
            results.append({
                "id": item["id"],
                "question": question,
                "category": category,
                "reference_answer": reference,
                "actual_answer": "",
                "steps": [],
                "correctness": 0,
                "groundedness": 0,
                "relevance": 0,
                "comment": f"Graph execution failed: {run_result['error']}",
            })
            time.sleep(PAUSE_BETWEEN_QUESTIONS)
            continue

        actual_answer = run_result["answer"]
        print(f"   Ответ: {actual_answer[:150]}")

        try:
            score = judge_answer(question, reference, actual_answer)
            print(f"   ✅ correctness={score.correctness} groundedness={score.groundedness} relevance={score.relevance}")
        except Exception as e:
            print(f"   ⚠️ Judge failed: {e}")
            score = JudgeScore(correctness=0, groundedness=0, relevance=0, comment=f"Judge failed: {e}")

        results.append({
            "id": item["id"],
            "question": question,
            "category": category,
            "reference_answer": reference,
            "actual_answer": actual_answer,
            "steps": run_result["steps"],
            "correctness": score.correctness,
            "groundedness": score.groundedness,
            "relevance": score.relevance,
            "comment": score.comment,
        })

        print()
        time.sleep(PAUSE_BETWEEN_QUESTIONS)

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("📊 ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 70)
    n = len(results)
    avg_correctness = sum(r["correctness"] for r in results) / n
    avg_groundedness = sum(r["groundedness"] for r in results) / n
    avg_relevance = sum(r["relevance"] for r in results) / n

    print(f"{'ID':<4}{'Category':<14}{'Correct':<9}{'Grounded':<10}{'Relevant':<10}")
    for r in results:
        print(f"{r['id']:<4}{r['category']:<14}{r['correctness']:<9}{r['groundedness']:<10}{r['relevance']:<10}")

    print("-" * 70)
    print(f"СРЕДНИЕ: correctness={avg_correctness:.2f}  groundedness={avg_groundedness:.2f}  relevance={avg_relevance:.2f}")
    print(f"\n✅ Полные результаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    main()