import os
import uuid

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example memory
memory_text = "I am learning Python and artificial intelligence."

# Generate embedding
embedding = model.encode(memory_text).tolist()

print("Embedding dimension:", len(embedding))

# Connect to PostgreSQL
connection = psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

# Create a test user
user_id = uuid.uuid4()

with connection.cursor() as cursor:
    cursor.execute(
        """
        INSERT INTO users (id)
        VALUES (%s)
        """,
        (user_id,),
    )

    # Insert the memory
    cursor.execute(
        """
        INSERT INTO memories (id, user_id, content, embedding)
        VALUES (%s, %s, %s, %s)
        """,
        (
            uuid.uuid4(),
            user_id,
            memory_text,
            embedding,
        ),
    )

connection.commit()
connection.close()

print("Memory stored successfully!")
