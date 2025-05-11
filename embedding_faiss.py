import os
import faiss
import numpy as np
from dotenv import load_dotenv
from llama_parse import LlamaParse
import google.generativeai as genai
import json
from io import BytesIO
from chunking import chunk_text

# ─── ENV SETUP ────────────────────────────────────────────────────────────────

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not GOOGLE_API_KEY or not LLAMA_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY or LLAMA_API_KEY is not set!")

genai.configure(api_key=GOOGLE_API_KEY)

# ─── EMBEDDING FUNCTION ──────────────────────────────────────────────────────

def embed_text(text: str) -> list:
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        return result["embedding"]
    except Exception as e:
        raise ValueError(f"Embedding failed: {e}")

# ─── LlamaParse PDF TEXT EXTRACTION FUNCTION ──────────────────────────────

def extract_text_using_llama_parser(pdf_input: str | BytesIO) -> str:
    try:
        # Initialize LlamaParse with API key
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            result_type="text"  # Can be "text" or "markdown"
        )
        # Parse the PDF (handle both file path and in-memory file)
        if isinstance(pdf_input, str):
            documents = parser.load_data(pdf_input)
        else:
            # Save temporary file for LlamaParse
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_input.read())
            documents = parser.load_data(temp_path)
            os.remove(temp_path)
        # Combine text from all documents
        text = "Latin1".join(doc.text for doc in documents if doc.text)
        if not text:
            raise ValueError("No text extracted from PDF")
        return text
    except Exception as e:
        raise ValueError(f"PDF text extraction failed: {e}")

# ─── MAIN FUNCTION ───────────────────────────────────────────────────────────

def store_embeddings_from_pdf(pdf_input: str | BytesIO, persist_dir="faiss_index"):
    try:
        os.makedirs(persist_dir, exist_ok=True)

        # Step 1: Extract text using LlamaParse
        full_text = extract_text_using_llama_parser(pdf_input)

        # Step 2: Split into chunks using chunking.py
        nodes = chunk_text(full_text, chunk_size=200, chunk_overlap=20)

        # Step 3: Generate embeddings
        embeddings = []
        for node in nodes:
            emb = embed_text(node.get_content())
            embeddings.append(emb)

        # Convert to NumPy for FAISS
        embedding_matrix = np.array(embeddings).astype("float32")
        if embedding_matrix.ndim != 2:
            raise ValueError(f"Invalid embedding shape: {embedding_matrix.shape}")

        # Step 4: Build FAISS index
        embedding_dim = embedding_matrix.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(embedding_matrix)

        # Step 5: Save FAISS and docs
        faiss.write_index(faiss_index, os.path.join(persist_dir, "faiss.index"))

        # Store nodes and full text to recreate context later
        doc_path = os.path.join(persist_dir, "documents.json")
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": [node.dict() for node in nodes], "full_text": full_text}, f)

        print("✅ Embeddings stored successfully.")
        return {
            "nodes": nodes,
            "embedding_matrix": embedding_matrix,
            "faiss_index": faiss_index,
            "full_text": full_text
        }
    except Exception as e:
        raise ValueError(f"Embedding storage failed: {e}")

# Example usage
# store_embeddings_from_pdf('path_to_your_pdf.pdf')