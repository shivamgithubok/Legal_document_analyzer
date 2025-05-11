import os
import json
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import faiss
import google.generativeai as genai
from embedding_faiss import embed_text, store_embeddings_from_pdf
from io import BytesIO

# ─── SETUP ─────────────────────────────────────────────────────────────────────

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is not set! Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
PERSIST_DIR = "faiss_index"
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(PERSIST_DIR, "documents.json")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ─── UI: PDF UPLOAD ───────────────────────────────────────────────────────────

st.title("📄 Legal Document Analyzer (RAG-Based)")

uploaded_file = st.file_uploader("📂 Upload a Legal Agreement (PDF)", type=["pdf"])
if uploaded_file:
    # Process PDF using embedding_faiss
    try:
        # Pass the uploaded file as a BytesIO object
        pdf_data = BytesIO(uploaded_file.read())
        result = store_embeddings_from_pdf(pdf_data, persist_dir=PERSIST_DIR)
        full_text = result["full_text"]
        nodes = result["nodes"]
        st.success("✅ PDF processed and embeddings stored successfully.")
    except Exception as e:
        st.error(f"❌ PDF processing failed: {e}")
        st.stop()

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

if os.path.exists(DOCSTORE_PATH) and st.button("📝 Generate Agreement Summary"):
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        full_text = data.get("full_text", "")

    if not full_text:
        st.error("❌ No text available for summarization.")
        st.stop()

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a legal expert. Summarize the following legal agreement in 150-200 words, "
            "focusing on key clauses, parties involved, and main obligations. Ensure the summary is clear and concise:\n\n"
            f"{full_text[:4000]}"  # Limit input to avoid API token limits
        )
        response = model.generate_content(prompt)
        summary = response.text
        st.subheader("📝 Agreement Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"❌ Summarization failed: {e}")

# ─── Q&A (RAG-BASED) ──────────────────────────────────────────────────────────

st.subheader("🔍 Ask a Question About the Agreement")
user_query = st.text_input("💬 Enter your question (e.g., 'What are the termination clauses?'):")
if user_query:
    if not os.path.exists(DOCSTORE_PATH) or not os.path.exists(INDEX_PATH):
        st.warning("⚠️ No document processed. Please upload and process a PDF first.")
    else:
        # Load documents and FAISS index
        try:
            with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            node_texts = [node["text"] for node in data["nodes"]]
            faiss_index = faiss.read_index(INDEX_PATH)
        except Exception as e:
            st.error(f"❌ Failed to load index or document store: {e}")
            st.stop()

        # Embed user query
        try:
            query_embedding = np.array(embed_text(user_query)).astype("float32").reshape(1, -1)
            if query_embedding.shape[1] != faiss_index.d:
                st.error(f"❌ Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {faiss_index.d}.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Query embedding failed: {e}")
            st.stop()

        # Perform FAISS search (RAG retrieval)
        try:
            D, I = faiss_index.search(query_embedding, k=3)  # Retrieve top 3 chunks
            valid_indices = [i for i in I[0] if i < len(node_texts) and D[0][list(I[0]).index(i)] < 1.0]  # L2 distance threshold
            if not valid_indices:
                st.warning("⚠️ No relevant information found in the document for this question.")
                st.stop()
            context = "\n\n".join([node_texts[i] for i in valid_indices])
        except Exception as e:
            st.error(f"❌ FAISS search failed: {e}")
            st.stop()

        # Generate answer using Gemini (RAG generation)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are a legal expert. Based on the following context from a legal agreement, "
                "provide a concise and accurate answer to the question. If the context doesn't contain "
                "enough information, say so clearly:\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_query}"
            )
            response = model.generate_content(prompt)
            answer = response.text
            st.subheader("💡 Answer")
            st.write(answer)

            # Option to view context
            with st.expander("📄 View Relevant Document Chunks"):
                st.write(context)
        except Exception as e:
            st.error(f"❌ Answer generation failed: {e}")