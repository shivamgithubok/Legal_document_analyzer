
# Query engine setup
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(index.as_retriever())

# Define a query
query_text = "example query related to your documents"
response = query_engine.query(query_text)

# Print results
print(f"✅ Query: {query_text}\n")
print("🔍 Most relevant documents:")
print(response)
