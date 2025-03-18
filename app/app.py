import os
import json
import faiss
import fitz  # PyMuPDF for PDF extraction
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Error: Gemini API key is missing. Set it in the .env file.")
    st.stop()

#  Configure Gemini AI
genai.configure(api_key=API_KEY)

#  Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index Path
FAISS_INDEX_PATH = "faiss_index/faiss.index"
DOCSTORE_PATH = "faiss_index/docstore.json"

#  Ensure paths exist
os.makedirs("faiss_index", exist_ok=True)

# Streamlit UI
st.title("📜 Legal Document Analyzer with AI")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a PDF File", type=["pdf"])

if uploaded_file:
    st.write(" PDF uploaded successfully!")

    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text_data = [page.get_text("text") for page in doc]
    
    #Combine all text for summarization
    full_text = "\n".join(text_data)

    #Generate a summary using Gemini AI
    try:
        summary_prompt = f"Summarize the following legal document:\n\n{full_text}"
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(summary_prompt)
        summary = response.text if hasattr(response, "text") else "Error: No text in response."
        st.subheader("📜 AI Summary of the Document:")
        st.write(summary)
    except Exception as e:
        st.error(f"❌ Error: {e}")

    # Break text into chunks for FAISS search
    chunk_size = 500
    document_chunks = []
    
    for text in text_data:
        words = text.split()
        for j in range(0, len(words), chunk_size):
            chunk = " ".join(words[j:j+chunk_size])
            document_chunks.append(chunk)

    st.write(f"📄 Extracted {len(document_chunks)} chunks for searching.")

    #Generate embeddings
    embeddings = embedding_model.encode(document_chunks)

    #Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    #Save FAISS Index
    faiss.write_index(index, FAISS_INDEX_PATH)

    #Save document store
    docstore = {str(i): document_chunks[i] for i in range(len(document_chunks))}
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(docstore, f)

    st.success("PDF processed and indexed successfully!")

    #Query Section
    st.subheader("🔍 Ask Questions About the Document")
    user_query = st.text_input("💬 Enter your query:")

    if user_query:
        query_embedding = embedding_model.encode([user_query])
        _, indices = index.search(query_embedding, 5)  # Retrieve top 5 chunks

        retrieved_texts = [docstore[str(idx)] for idx in indices[0] if str(idx) in docstore]

        if retrieved_texts:
            # Answer user query using Gemini
            prompt = f"Based on the following legal document, answer: {user_query}\n\n" + "\n".join(retrieved_texts)

            try:
                response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
                answer = response.text if hasattr(response, "text") else "Error: No text in response."
                st.subheader("💡")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ No relevant documents found!")
