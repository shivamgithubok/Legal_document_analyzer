import faiss
import numpy as np
from llama_index.core import StorageContext, load_index_from_storage

# Example: Recreate FAISS index and save it
def build_faiss_index():
    d = 384  # Dimension (match SentenceTransformer model output size)
    index = faiss.IndexFlatL2(d)

    # Example embeddings (random vectors, replace with actual ones)
    data = np.random.rand(100, d).astype('float32')  # 100 dummy vectors
    index.add(data)

    # Save FAISS index
    faiss.write_index(index, "faiss_index/index.faiss")
    print("✅ FAISS index rebuilt and saved!")

build_faiss_index()
