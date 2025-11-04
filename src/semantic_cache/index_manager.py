import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.logger.logger import get_logger
from src.semantic_cache.embedder import get_embedder
from pathlib import Path


logger = get_logger(__name__)

# Paths
DEFAULT_INDEX_DIR = "data/cache_store"  
DEFAULT_INDEX_NAME = "semantic_cache"


def save_faiss_index(vector_store: FAISS, index_dir: str, index_name: str):
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(folder_path=index_dir, index_name=index_name)
    logger.info(f"Saved FAISS index to {index_dir}/{index_name}.faiss")

def create_faiss_index(
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME
) -> FAISS:
    embedder = get_embedder()

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas
    )
    logger.info("[index_manager] Created FAISS index")

    save_faiss_index(vector_store, index_dir, index_name)
    return vector_store

def load_faiss_index(
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu"
) -> Optional[FAISS]:
    embedder = get_embedder(model_name, device)
    index_path = Path(index_dir) / f"{index_name}.faiss"

    if not index_path.exists():
        logger.warning(f"Index not found at {index_path}")
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS index from {index_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None

def reset_faiss_index(index_dir: str = DEFAULT_INDEX_DIR):
    """
    Deletes all FAISS index files (use with caution).
    """
    for file in Path(index_dir).glob("*"):
        file.unlink()
    # print(f"[index_manager] Reset FAISS index in {index_dir}")
    logger.info(f"[index_manager] Reset FAISS index in {index_dir}")


if __name__ == "__main__":
    texts = [
        "What is the capital of France?",
        "How does semantic caching work?",
        "Applications of machine learning in healthcare"
    ]
