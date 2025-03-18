import os
import pandas as pd
import numpy as np
import faiss
import pickle
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Initialize Embedding Model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

def preprocess_and_store_faiss(casedf, faiss_index_file="faiss_index.pkl"):
    """Processes legal text, chunks it, embeds it, and stores in FAISS"""
    
    # ✅ Preprocess Data
    casedf.dropna(how="all", inplace=True)
    casedf["Author"] = casedf["Author"].fillna("Unknown")
    casedf["Name"] = casedf["Name"].fillna("Unknown")
    casedf["OpinionText"] = casedf["OpinionText"].fillna("No opinion available")
    casedf["DecisionDate"] = pd.to_datetime(casedf["DecisionDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    casedf["OpinionText"] = casedf["OpinionText"].str.replace(r"[\n\t]", " ", regex=True).str.strip()

    # ✅ Sample only 1/32 for efficiency
    sampled_df = casedf.sample(frac=1/32, random_state=42)

    # ✅ Combine text into structured format
    text_data = []
    for _, row in sampled_df.iterrows():
        text_data.append(f"Case: {row['Name']}\nAuthor: {row['Author']}\nCitation: {row['Citation']}\nDate: {row['DecisionDate']}\nType: {row['OpinionType']}\nText: {row['OpinionText']}\n")
    
    full_text = "\n\n".join(text_data)

    # ✅ Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_text(full_text)

    # ✅ Convert Chunks to Documents
    documents = [Document(text=chunk) for chunk in chunks]

    # ✅ Initialize FAISS
    embedding_dim = len(embed_model.get_text_embedding("sample text"))
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ✅ Embed and Store in FAISS
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    # ✅ Save FAISS index
    with open(faiss_index_file, "wb") as f:
        pickle.dump((faiss_index, chunks), f)

    print(f"✅ FAISS Index successfully saved at: {faiss_index_file}")
    print(f"🔹 Total Documents Embedded: {len(chunks)}")
