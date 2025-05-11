import os
import json
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import faiss
import google.generativeai as genai
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

# ─── SETUP ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "legal_document_analyzer_secret_key"
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is not set! Please add it to your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
PERSIST_DIR = "faiss_index"
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(PERSIST_DIR, "documents.json")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Embedding function using Google Gemini
def embed_text(text: str) -> list:
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        return result["embedding"]
    except Exception as e:
        raise Exception(f"Embedding failed: {e}")

# Process PDF and store embeddings
def process_pdf(file_path):
    try:
        # Extract text from PDF
        doc = fitz.open(file_path)
        pages = [page.get_text("text") for page in doc]
        full_text = "\n".join(pages).strip()
        if not full_text:
            raise ValueError("Could not extract text from the PDF.")

        # Chunk the document
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        docs = [Document(text=full_text)]
        nodes = parser.get_nodes_from_documents(docs)
        if not nodes:
            raise ValueError("No chunks created from the document.")

        # Compute embeddings
        embeddings = [embed_text(node.get_content()) for node in nodes]
        embedding_matrix = np.array(embeddings).astype("float32")
        if embedding_matrix.ndim != 2:
            raise ValueError("Invalid embedding shape. Expected 2D array.")
        embedding_dim = embedding_matrix.shape[1]

        # Build or update FAISS index
        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
            if index.d != embedding_dim:
                raise ValueError(f"Dimension mismatch! Existing index dim: {index.d}, new embedding dim: {embedding_dim}")
            index.add(embedding_matrix)
        else:
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embedding_matrix)

        # Save index and document store
        faiss.write_index(index, INDEX_PATH)
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"nodes": [node.dict() for node in nodes], "full_text": full_text}, f)
        
        return True, full_text
    except Exception as e:
        return False, str(e)

# Generate summary
def generate_summary(full_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a legal expert. Summarize the following legal agreement in 150-200 words, "
            "focusing on key clauses, parties involved, and main obligations. Ensure the summary is clear and concise:\n\n"
            f"{full_text[:4000]}"  # Limit input to avoid API token limits
        )
        response = model.generate_content(prompt)
        return True, response.text
    except Exception as e:
        return False, f"Summarization failed: {e}"

# Answer question
def answer_question(query):
    try:
        if not os.path.exists(DOCSTORE_PATH) or not os.path.exists(INDEX_PATH):
            return False, "No document processed. Please upload a PDF first.", None

        # Load documents and FAISS index
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        node_texts = [node["text"] for node in data["nodes"]]
        faiss_index = faiss.read_index(INDEX_PATH)

        # Embed user query
        query_embedding = np.array(embed_text(query)).astype("float32").reshape(1, -1)
        if query_embedding.shape[1] != faiss_index.d:
            return False, f"Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {faiss_index.d}.", None

        # Perform FAISS search
        D, I = faiss_index.search(query_embedding, k=3)  # Retrieve top 3 chunks
        valid_indices = [i for i in I[0] if i < len(node_texts) and D[0][list(I[0]).index(i)] < 1.0]  # L2 distance threshold
        if not valid_indices:
            return False, "No relevant information found in the document for this question.", None
        context = "\n\n".join([node_texts[i] for i in valid_indices])

        # Generate answer
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a legal expert. Based on the following context from a legal agreement, "
            "provide a concise and accurate answer to the question. If the context doesn't contain "
            "enough information, say so clearly:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        response = model.generate_content(prompt)
        return True, response.text, context
    except Exception as e:
        return False, f"Answer generation failed: {e}", None

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    answer = None
    context = None
    error = None

    if request.method == "POST":
        # Handle PDF upload
        if "pdf_file" in request.files:
            file = request.files["pdf_file"]
            if file and file.filename.endswith(".pdf"):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                success, result = process_pdf(file_path)
                if success:
                    flash("✅ PDF processed successfully!", "success")
                    os.remove(file_path)  # Clean up uploaded file
                else:
                    flash(f"❌ {result}", "error")
                    error = result

        # Handle summary request
        elif "generate_summary" in request.form:
            if os.path.exists(DOCSTORE_PATH):
                with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
                    full_text = json.load(f).get("full_text", "")
                success, result = generate_summary(full_text)
                if success:
                    summary = result
                    flash("✅ Summary generated successfully!", "success")
                else:
                    flash(f"❌ {result}", "error")
                    error = result

        # Handle question
        elif "question" in request.form:
            query = request.form.get("question")
            if query:
                success, result, ctx = answer_question(query)
                if success:
                    answer = result
                    context = ctx
                    flash("✅ Question answered successfully!", "success")
                else:
                    flash(f"❌ {result}", "error")
                    error = result

    return render_template("index.html", summary=summary, answer=answer, context=context, error=error)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable reloader to avoid signal error