from sentence_transformers import SentenceTransformer

model_name = "nlpaueb/legal-bert-base-uncased"
model = SentenceTransformer(model_name)

# Save the model locally to avoid re-downloading each time
model.save("legal-bert")
print("Model downloaded and saved successfully!")
