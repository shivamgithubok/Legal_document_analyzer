import os
import faiss
import torch
import json
import streamlit as st
import google.generativeai as genai
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ Error: Gemini API key is missing. Set it in the .env file.")
    st.stop()

# ✅ Configure Gemini AI
genai.configure(api_key=API_KEY)

# ✅ Paths
persist_dir = "faiss_index"
faiss_path = os.path.join(persist_dir, "faiss.index")

# ✅ Load Legal-BERT from local directory
model_path = "legal-bert"  
if not os.path.exists(model_path):
    st.error(f"❌ Error: Local model directory '{model_path}' not found!")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name=model_path, device=device) 

# ✅ Load FAISS index
if not os.path.exists(faiss_path):
    st.error(f"❌ Error: FAISS index not found at {faiss_path}")
    st.stop()

dimension = 768  # Legal-BERT embeddings have 768 dimensions
faiss_index = faiss.read_index(faiss_path)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# ✅ Load index from storage
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = load_index_from_storage(storage_context, embed_model=embed_model)

# ✅ Load document store (to retrieve original texts)
docstore_path = os.path.join(persist_dir, "docstore.json")
if not os.path.exists(docstore_path):
    st.error(f"❌ Error: Document store not found at {docstore_path}")
    st.stop()

with open(docstore_path, "r", encoding="utf-8") as f:
    docstore = json.load(f)

# ✅ Streamlit UI
st.title("📜 Legal Document Search with AI")

# ✅ User Query Input
user_query = st.text_input("🔍 Enter your query:")

if user_query:
    # Retrieve relevant chunks using LlamaIndex
    query_engine = index.as_query_engine()
    response = query_engine.query(user_query)

    if response.response:
        # Summarize retrieved text using Gemini AI
        prompt = f"Summarize the following legal text:\n\n{response.response}"
        
        try:
            gemini_response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
            summary = gemini_response.text if hasattr(gemini_response, "text") else "Error: No text in response."
            st.subheader("📜 AI Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ No relevant documents found!")
