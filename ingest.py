import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "analyst_docs"

# Абсолютный путь: работает независимо от того, откуда запущен скрипт
# (main.py из корня проекта ИЛИ eval/evaluate.py из подпапки eval/).
PROJECT_ROOT = Path(__file__).resolve().parent
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_db")

EMBED_DIM = 3072  # дефолтная размерность gemini-embedding-001

_client = None
_vectorstore = None


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def get_vectorstore():
    """Возвращает singleton QdrantVectorStore, создавая коллекцию при первом вызове."""
    global _vectorstore
    if _vectorstore is None:
        client = _get_client()
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vectorstore


def ingest_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"✅ Загружено {len(chunks)} чанков из {file_path}")


if __name__ == "__main__":
    ingest_documents(str(PROJECT_ROOT / "docs" / "company_info.txt"))