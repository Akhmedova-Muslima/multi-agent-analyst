# Multi-Agent AI Analyst

Мультиагентная система на LangGraph: supervisor маршрутизирует вопрос к нужному
специалисту (retriever / web / data-SQL / code), critic проверяет финальный ответ
и запускает ревизию при необходимости. Полностью на бесплатных тарифах
(Gemini + Qdrant embedded + Tavily + Langfuse), без карты.

**Live demo:** `(https://multi-agent-analyst-navy.vercel.app/)`
**Backend API:** `(https://multi-agent-analyst-kztk.onrender.com)`

---

## Архитектура

```
                    Question
                       │
                       ▼
              ┌─────────────────┐
        ┌────▶│   Supervisor    │◀────┐
        │     │  (router, LLM)  │     │
        │     └────────┬────────┘     │
        │              │              │
   ┌────┴────┬─────────┼─────────┬────┴────┐
   ▼         ▼          ▼         ▼         │
Retriever   Web       Data      Code        │
 (RAG)   (Tavily)   (SQL) ───▶ (Python)     │
   │         │          │         │         │
   └─────────┴──────────┴─────────┘         │
                       │ (finish)            │
                       ▼                     │
                  ┌──────────┐               │
                  │ Generate │               │
                  └────┬─────┘               │
                       ▼                     │
                  ┌──────────┐   revise      │
                  │  Critic  │───────────────┘
                  └────┬─────┘
                       │ approved
                       ▼
                    Answer
```

<img width="933" height="777" alt="image" src="https://github.com/user-attachments/assets/066fb00d-eb0f-4dee-b8fd-803628491ec8" />


## Стек

| Компонент | Технология |
|---|---|
| Оркестрация | LangGraph (StateGraph, conditional edges) |
| LLM | Gemini (`gemini-3.1-flash-lite`) |
| Векторная БД | Qdrant (embedded) |
| SQL | SQLite, read-only guard (только `SELECT`) |
| Код-агент | sandboxed `exec`, cap 10с (`func_timeout`) |
| Веб-поиск | Tavily |
| Память | отдельная Qdrant-коллекция `conversation_memory` |
| Оценка | LLM-as-judge (correctness / groundedness / relevance, 1-5) |
| Трейсинг | Langfuse |
| Backend | FastAPI + SSE (`/chat`) |
| Frontend | Next.js, живая SVG-визуализация графа |
| Деплой | Render (backend, Docker) + Vercel (frontend) |

## Запуск локально

```powershell
# backend
pip install -r requirements.txt
python setup_db.py        # если БД ещё не создана
uvicorn api_server:app --reload --port 8000

# frontend (в отдельном терминале)
cd frontend
npm install
npm run dev
```


## Эволюция и отладка (что было исправлено)

В процессе разработки было обнаружено и исправлено 3 конкретных бага —
хороший пример того, зачем нужен evaluation harness: без него эти ошибки
остались бы незамеченными, ответы выглядели правдоподобно на первый взгляд.

**1. Supervisor неверно роутил вопросы с пересекающимися ключевыми словами.**
Вопрос "Какая политика возврата средств?" содержит и generic-слово "какая"
(из `DATA_KEYWORDS`), и специфичное "возврат"/"политик" (из
`RETRIEVER_KEYWORDS`). Приоритет был отдан generic-словам → граф шёл в
`data` вместо `retriever`, ответ: *"документы не содержат информации"*
(correctness=1). **Фикс**: убран guard `and not wants_data`, специфичные
ключевые слова теперь приоритетнее generic-вопросительных местоимений.

**2. SQL-агент не включал числовую колонку в SELECT для extreme-value
вопросов.** На вопрос "какой квартал был самым прибыльным" LLM генерировал
`SELECT quarter FROM metrics ORDER BY revenue DESC LIMIT 1` — без
`revenue` в SELECT. Ответ терял число (*"Самым прибыльным был Q4"* вместо
*"Q4 с доходом 610000"*), judge снижал correctness за неполноту.
**Фикс**: в промпт `data_agent` добавлено явное требование включать
сравниваемую числовую колонку в SELECT при вопросах на "максимум/минимум".

**3. Retriever не находил релевантный чанк при k=4.** Для вопроса про
политику возврата top-4 similarity search не дотягивался до нужного куска
документа, хотя тот же документ находился нормально для соседних вопросов.
**Фикс**: `k` увеличен с 4 до 6.

После всех трёх фиксов — 11/11 вопросов, оба варианта (с critic и без),
correctness/groundedness/relevance = 5.00/5.00/5.00.

## Evaluation — метрики (LLM-as-judge, 1-5)

Экономный LLM-judge вместо полного RAGAS — на бесплатной квоте Gemini
(15 запросов/мин) полный RAGAS (faithfulness + answer_relevancy +
context_precision, 3-5 доп. вызовов на метрику) гарантированно упирается
в дневной лимит на 11 вопросах. Judge покрывает те же три оси качества
одним вызовом на вопрос.

### С critic vs без critic (`eval/compare_critic.py`)

| Метрика | С critic | Без critic | Δ |
|---|---|---|---|
| correctness | 5.00 | 5.00 | +0.00 |
| groundedness | 5.00 | 5.00 | +0.00 |
| relevance | 5.00 | 5.00 | +0.00 |

**Интерпретация**: на текущем тестсете (11 вопросов, после фиксов роутинга/SQL/
retriever) обе конфигурации дают идентичный результат — critic не находит,
что исправлять, потому что специалисты уже отвечают корректно. Роль critic
здесь — safety net, а не источник видимого прироста качества на этом наборе:
его ценность подтверждена отдельно (см. F8) на намеренно некорректном ответе,
где он поймал ошибку и инициировал ревизию. На более разнообразном/шумном
тестсете ожидаемо появилась бы измеримая дельта.
