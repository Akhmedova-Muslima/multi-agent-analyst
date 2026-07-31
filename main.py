from graph import app
from memory import save_turn
from observability import get_invoke_config

if __name__ == "__main__":
    question = input("Вопрос: ")
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
    result = app.invoke(initial_state, config=get_invoke_config(recursion_limit=40))
    print("\n--- Путь агентов ---")
    print(" → ".join(result["steps"]))
    print("\nОтвет:", result["answer"])

    save_turn(question, result["answer"])