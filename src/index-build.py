import polars as pl
import faiss, pickle, os, re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src import config

def build_system():
    print("🛠️ TASK 1: Preprocessing 9.6M Records...")
    df = pl.read_csv(config.RAW_DATA_PATH, ignore_errors=True)
    
    # Cleaning
    target_products = ['Credit card', 'Credit card or prepaid card', 'Consumer Loan', 
                       'Checking or savings account', 'Money transfer']
    
    df = df.filter(pl.col("Consumer complaint narrative").is_not_null())
    df = df.filter(pl.col("Product").is_in(target_products))
    
    def clean(t):
        t = t.lower()
        t = re.sub(r'x{2,}|[0-9/]', '', t)
        return " ".join(t.split())

    df = df.with_columns(pl.col("Consumer complaint narrative").map_elements(clean).alias("cleaned"))
    
    print("🛠️ TASK 2: Stratified Indexing...")
    # True Stratification
    sample = df.group_by("Product").map_groups(lambda g: g.sample(n=min(len(g), config.SAMPLE_PER_CAT), seed=42))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks, meta = [], []
    for row in sample.to_dicts():
        d_chunks = splitter.split_text(row['cleaned'])
        for c in d_chunks:
            chunks.append(c)
            meta.append({"product": row['Product'], "complaint_id": row.get('Complaint ID')})

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    os.makedirs(config.INDEX_PATH.parent, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": meta}, f)
    print("✅ Build Complete.")

if __name__ == "__main__":
    build_system()