import polars as pl
import faiss, pickle, os, re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src import config

def run_build():
    print("Task 2: Starting visible index build...")
    df = pl.read_csv(config.PROCESSED_DATA_PATH)
    sample = df.group_by("product_category").map_groups(lambda g: g.sample(n=min(len(g), 3000), seed=42))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks, meta = [], []
    for row in sample.to_dicts():
        d_chunks = splitter.split_text(str(row['cleaned_narrative']))
        for c in d_chunks:
            chunks.append(c)
            meta.append({"product": row['product_category'], "complaint_id": row.get('Complaint ID')})
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": meta}, f)
    print("✅ Index Build Complete.")
if __name__ == "__main__":
    run_build()
