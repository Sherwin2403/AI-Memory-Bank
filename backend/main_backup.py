import os
import uuid

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from backend.neo4j_client import create_memory_graph

load_dotenv()

app = FastAPI(title="AI Memory Bank")

# Load embedding model once when the API starts
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class MemoryRequest(BaseModel):
    user_id: str
    content: str


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/neo4j")
def neo4j_health():
    return {"status": "Neo4j client configured"}

@app.post("/memory")
def create_memory(memory: MemoryRequest):
    memory_id = str(uuid.uuid4())

    # Generate 384-dimensional embedding
    embedding = model.encode(memory.content).tolist()

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memories (
                    id,
                    user_id,
                    content,
                    metadata,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::vector
                )
                """,
                (
                    memory_id,
                    memory.user_id,
                    memory.content,
                    "{}",
                    embedding,
                ),
            )

        connection.commit()
        create_memory_graph(
            user_id=memory.user_id,
            memory_id=memory_id,
            content=memory.content,
        )

    finally:
        connection.close()

    return {
        "id": memory_id,
        "content": memory.content,
        "message": "Memory stored successfully",
    }
class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 5


@app.post("/memory/search")
def search_memories(request: MemorySearchRequest):
    query_embedding = model.encode(request.query).tolist()

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    metadata,
                    created_at,
                    access_count,
                    memory_weight,
                    embedding <=> %s::vector AS distance
                FROM memories
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    request.user_id,
                    query_embedding,
                    request.limit,
                ),
            )

            rows = cursor.fetchall()

            memories = []

            for row in rows:
                (
                    memory_id,
                    content,
                    metadata,
                    created_at,
                    access_count,
                    memory_weight,
                    distance,
                ) = row

                cursor.execute(
                    """
                    UPDATE memories
                    SET
                        access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (memory_id,),
                )

                memories.append(
                    {
                        "id": str(memory_id),
                        "content": content,
                        "metadata": metadata,
                        "created_at": created_at.isoformat(),
                        "access_count": access_count + 1,
                        "memory_weight": memory_weight,
                        "distance": float(distance),
                    }
                )

        connection.commit()

    finally:
        connection.close()

    return {
        "query": request.query,
        "results": memories,
    }
