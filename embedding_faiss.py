import os
import faiss
import torch
from llama_index.core import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

# ✅ Ensure FAISS directory exists
persist_dir = "faiss_index"
os.makedirs(persist_dir, exist_ok=True)

# ✅ Load Legal-BERT embedding model
model_name = "nlpaueb/legal-bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Loading model: {model_name} on {device.upper()}")

embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)

# ✅ Read file content safely
file_path = "chunked_output.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Error: File '{file_path}' not found!")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().strip()
    if not text:
        raise ValueError(f"❌ Error: File '{file_path}' is empty!")

# ✅ Create document object
documents = [Document(text=text)]

# ✅ Chunking for better retrieval
parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

# ✅ FAISS Initialization (Only CPU for Colab)
dimension = 768  # Legal-BERT outputs 768-dimensional vectors
faiss_index = faiss.IndexFlatL2(dimension)  # FAISS CPU version

vector_store = FaissVectorStore(faiss_index=faiss_index)

# ✅ Create and save index with embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# ✅ Persist metadata
index.storage_context.persist(persist_dir=persist_dir)

# ✅ Save FAISS index separately
faiss_path = os.path.join(persist_dir, "faiss.index")
faiss.write_index(faiss_index, faiss_path)

print(f"✅ FAISS index and metadata saved successfully on {device.upper()}!")
