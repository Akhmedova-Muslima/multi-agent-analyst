from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client.models import Distance, VectorParams
from ingest import _get_client, EMBED_DIM, EMBED_MODEL

MEMORY_COLLECTION = "conversation_memory"

_memory_store = None


def get_memory_store():
    """Отдельная коллекция Qdrant для истории диалогов (F10)."""
    global _memory_store
    if _memory_store is None:
        client = _get_client()
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

        if not client.collection_exists(MEMORY_COLLECTION):
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

        _memory_store = QdrantVectorStore(
            client=client,
            collection_name=MEMORY_COLLECTION,
            embedding=embeddings,
        )
    return _memory_store


def save_turn(question: str, answer: str):
    """Сохраняет пару вопрос-ответ в долгосрочную память."""
    try:
        store = get_memory_store()
        text = f"Q: {question}\nA: {answer}"
        store.add_documents([Document(page_content=text)])
    except Exception as e:
        print(f"⚠️ Не удалось сохранить в память: {e}")


def recall_relevant(question: str, k: int = 3) -> list[str]:
    """Достаёт релевантные прошлые ходы для нового вопроса.
    В случае любой ошибки молча возвращает пустой список,
    чтобы не засорять промпт супервизора мусорным текстом ошибки."""
    try:
        store = get_memory_store()
        docs = store.similarity_search(question, k=k)
        return [d.page_content for d in docs]
    except Exception as e:
        print(f"⚠️ Память недоступна: {e}")
        return []