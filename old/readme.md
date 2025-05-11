# Legal Document Analyzer

## 📌 Overview
This project is a **Legal Document Analyzer** that efficiently retrieves legal documents using **FAISS (Facebook AI Similarity Search) vector indexing** and **Legal-BERT embeddings**. It is designed to enhance legal research by enabling fast, accurate, and scalable document retrieval.

## 🚀 Features
✅ **Legal-BERT Embeddings**: Uses **nlpaueb/legal-bert-base-uncased** to generate domain-specific vector embeddings.
✅ **FAISS Vector Search**: Implements FAISS for fast nearest neighbor searches on large datasets.
✅ **Chunking & Preprocessing**: Documents are split into manageable chunks for better retrieval.
✅ **Efficient Storage & Retrieval**: Uses **LlamaIndex** for document structuring and FAISS for indexing.
✅ **Scalable & High-Performance**: Optimized for large-scale legal document databases.

## 📂 Project Structure
```
📦 Legal Document Analyzer
├── 📄 google_embed.py   # Main script to create FAISS index
├── 📄 query_engine.py   # Search query implementation
├── 📄 requirements.txt  # Dependencies
├── 📂 faiss_index       # Stored FAISS index files
├── 📄 README.md         # Project documentation
└── 📄 chunked_output.txt # Preprocessed legal document data
```

## 🛠️ Techniques Used
### **1. Text Embeddings with Legal-BERT**
We use **Legal-BERT** to convert legal text into numerical vectors. Unlike generic embeddings, **Legal-BERT** understands legal terminologies and context better.

### **2. Document Chunking**
Since legal documents are often long, we break them into **fixed-size chunks (200 tokens, 20 overlapping)** to improve retrieval accuracy.

### **3. FAISS for Vector Search**
FAISS enables efficient **Approximate Nearest Neighbor (ANN) search**, making it possible to retrieve similar legal documents in real time.

### **4. LlamaIndex for Storage Management**
We integrate **LlamaIndex** to structure, process, and manage the indexed data, ensuring smooth storage and retrieval.

## 🔁 Data Flow
1️⃣ **Text Processing** → Read legal documents and convert them into text chunks.
2️⃣ **Embedding Generation** → Convert chunks into vector representations using Legal-BERT.
3️⃣ **Indexing with FAISS** → Store embeddings in FAISS for fast similarity search.
4️⃣ **Query Processing** → Convert search queries into embeddings and retrieve the most relevant legal text from FAISS.

## 🔧 Installation & Usage
### **Step 1: Install Dependencies**
```
pip install -r requirements.txt
```

### **Step 2: Run Indexing Script**
This script reads legal documents, creates embeddings, and stores them in FAISS.
```
python google_embed.py
```

### **Step 3: Search for Legal Text**
Use `query_engine.py` to search for relevant legal documents.
```
python query_engine.py "What are the legal implications of contract breaches?"
```

## 🎯 Future Enhancements
🚀 Add **metadata tagging** for better classification.
🚀 Improve **query performance** with hybrid search (FAISS + BM25).
🚀 Extend to support **multi-language legal texts**.

## 🤝 Contributions & Feedback
We welcome feedback and contributions! Feel free to create an issue or pull request. Let's improve AI-powered legal research together! 🚀

#LegalTech #AI #FAISS #LegalBERT #MachineLearning #NLP #LlamaIndex #DataScience