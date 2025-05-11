# 📚 Legal Document Analyzer

## 📌 Overview
Legal Document Analyzer is a powerful tool that uses a Retrieval-Augmented Generation (RAG) pipeline to process legal agreements in PDF format. It combines **Google Gemini embeddings**, **FAISS indexing**, and **LlamaParse** for accurate text extraction. Users can generate concise summaries and ask specific legal questions like _"What are the termination clauses?"_ via a user-friendly **Streamlit** interface.

---

## 🚀 Features

- ✅ **Document Summarization**: Generates 150–200 word summaries with key clauses, parties, and obligations.
- ✅ **Interactive Q&A**: Allows precise question answering using relevant document chunks via RAG.
- ✅ **Google Gemini Embeddings**: Uses `models/embedding-001` for semantic similarity and `gemini-1.5-flash` for generation.
- ✅ **FAISS Vector Search**: Provides fast nearest-neighbor search.
- ✅ **LlamaParse Text Extraction**: High-accuracy PDF text extraction.
- ✅ **Chunking & Preprocessing**: Splits documents into 200-token chunks with 20-token overlap.
- ✅ **Streamlit UI**: Simple interface to upload PDFs, view summaries, and ask questions.
- ✅ **Robust Error Handling**: Ensures clear, reliable feedback on issues.

---

## 📂 Project Structure

📦 Legal Document Analyzer
├── 📄 chunking.py # Text chunking logic
├── 📄 embedding_faiss.py # Embedding + FAISS index generation
├── 📄 app.py # Streamlit UI + RAG pipeline
├── 📄 README.md # Documentation
├── 📄 requirements.txt # Required Python packages
├── 📂 faiss_index # Stores FAISS index + document metadata
└── 📄 .env # API keys & environment config



---

## 🛠️ Techniques Used

1. **Text Extraction with LlamaParse**  
   Extracts structured legal text from complex PDFs.

2. **Document Chunking**  
   Uses `SimpleNodeParser` to chunk text (200 tokens with 20-token overlap).

3. **Google Gemini Embeddings**  
   Embeds text using Gemini’s `embedding-001`; generates text with `gemini-1.5-flash`.

4. **FAISS for Vector Search**  
   Efficient ANN search to retrieve top 3 relevant chunks based on L2 distance.

5. **RAG Pipeline**  
   Combines FAISS-based retrieval with Gemini generation for accurate answers.

6. **Streamlit for UI**  
   Allows PDF upload, summarization, and Q&A via interactive browser app.

---

## 🔁 Data Flow

1️⃣ **PDF Processing** → Extract text with LlamaParse  
2️⃣ **Text Chunking** → Chunk using `chunking.py`  
3️⃣ **Embedding Generation** → Use Gemini to convert chunks into embeddings  
4️⃣ **Indexing with FAISS** → Store and search using FAISS (`embedding_faiss.py`)  
5️⃣ **Summary Generation** → Summarize document (`app.py`)  
6️⃣ **Query Processing** → Embed query, retrieve chunks, generate answer (`app.py`)  

---

## 🔧 Installation & Usage

### Step 1: Install Dependencies

Create a `requirements.txt`:
```txt
streamlit
llama-parse
llama-index-core
faiss-cpu
google-generativeai
python-dotenv
numpy



pip install -r requirements.txt


GOOGLE_API_KEY=your_gemini_api_key
LLAMA_API_KEY=your_llama_api_key


streamlit run app.py


---

Would you like me to save this to a `README.md` file for you?