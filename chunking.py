from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read the preprocessed text file
input_txt_path = "output_chunks.txt"
with open(input_txt_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Initialize RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust chunk size as needed
    chunk_overlap=50,  # Ensure slight overlap for context preservation
    separators=["\n\n", "\n", " ", ""],  # Order of splitting
)

# Create chunks
chunks = text_splitter.split_text(text_data)

# Save chunks to a new file
chunked_txt_path = "chunked_output.txt"
with open(chunked_txt_path, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"Chunk {i+1}:\n{chunk}\n")
        file.write("=" * 50 + "\n\n")

print(f"✅ Successfully split and saved {len(chunks)} chunks to {chunked_txt_path}")
