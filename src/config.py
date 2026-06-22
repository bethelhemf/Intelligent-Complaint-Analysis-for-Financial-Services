import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "complaints.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "filtered_complaints.csv"
INDEX_PATH = BASE_DIR / "vector_store" / "complaints_index.faiss"
METADATA_PATH = BASE_DIR / "vector_store" / "metadata.pkl"

# Model Selection
LLM_MODEL = "llama-3.1-8b-instant" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG Params
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SAMPLE_PER_CAT = 3000