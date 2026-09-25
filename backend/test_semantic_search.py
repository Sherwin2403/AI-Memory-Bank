import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query_text = "What programming and AI skills am I currently studying?"

query_embedding = model.encode(query_text).tolist()

connection = psycopg.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

with connection.cursor() as cursor:

    # Find the most semantically similar memory
    cursor.execute(
        """
        SELECT
            id,
            content,
            embedding <=> %s::vector AS distance
        FROM memories
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
        """,
        (query_embedding, query_embedding),
    )

    result = cursor.fetchone()

    if result is None:
        print("No memories found.")
    else:
        memory_id, content, distance = result

        print("Retrieved memory:")
        print(content)
        print(f"Distance: {distance:.4f}")

        # Record that this memory was actually used
        cursor.execute(
            """
            UPDATE memories
            SET
                access_count = access_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE id = %s;
            """,
            (memory_id,),
        )

        print("Memory access recorded.")

connection.commit()
connection.close()
