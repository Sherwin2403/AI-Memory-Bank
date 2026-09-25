from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding dimension:", model.get_embedding_dimension())

text = "I am learning Python and artificial intelligence."
embedding = model.encode(text)

print("Embedding shape:", embedding.shape)
