import os
import uuid

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from backend.concept_extractor import extract_concepts

from backend.neo4j_client import (
    create_memory_graph,
    create_concept_graph,
    update_memory_graph,
    delete_memory_graph,
)
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
        concepts = extract_concepts(memory.content)

        create_concept_graph(
            memory_id=memory_id,
            concepts=concepts,
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
                    last_accessed,
                    embedding <=> %s::vector AS distance
                FROM memories
                WHERE user_id = %s
                ORDER BY (
                    (1 - (embedding <=> %s::vector)) * 0.7
                    + memory_weight * 0.2
                    + LEAST(access_count, 10) / 10.0 * 0.1
                ) DESC
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
                    last_accessed,
                    distance,
                ) = row

                new_access_count = access_count + 1

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

                similarity = 1 - float(distance)

                score = (
                    similarity * 0.7
                    + float(memory_weight) * 0.2
                    + min(new_access_count, 10) / 10.0 * 0.1
                )

                memories.append(
                    {
                        "id": str(memory_id),
                        "content": content,
                        "metadata": metadata,
                        "created_at": created_at.isoformat(),
                        "access_count": new_access_count,
                        "memory_weight": memory_weight,
                        "distance": float(distance),
                        "similarity": similarity,
                        "score": score,
                    }
                )

        connection.commit()

    finally:
        connection.close()

    return {
        "query": request.query,
        "results": memories,
    }
class MemoryWeightRequest(BaseModel):
    memory_weight: float


@app.patch("/memory/{memory_id}/weight")
def update_memory_weight(
    memory_id: str,
    request: MemoryWeightRequest
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET memory_weight = %s
                WHERE id = %s
                RETURNING id, content, memory_weight
                """,
                (
                    request.memory_weight,
                    memory_id,
                ),
            )

            row = cursor.fetchone()

            if row is None:
                return {
                    "error": "Memory not found"
                }

        connection.commit()

    finally:
        connection.close()

    return {
        "id": str(row[0]),
        "content": row[1],
        "memory_weight": row[2],
        "message": "Memory weight updated successfully",
    }
class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    metadata: dict | None = None
    memory_weight: float | None = None


@app.patch("/memory/{memory_id}")
def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            # Get the existing memory
            cursor.execute(
                """
                SELECT content, metadata, memory_weight
                FROM memories
                WHERE id = %s
                """,
                (memory_id,),
            )

            existing = cursor.fetchone()

            if existing is None:
                return {"error": "Memory not found"}

            old_content, old_metadata, old_weight = existing

            new_content = (
                request.content
                if request.content is not None
                else old_content
            )

            new_metadata = (
                request.metadata
                if request.metadata is not None
                else old_metadata
            )

            new_weight = (
                request.memory_weight
                if request.memory_weight is not None
                else old_weight
            )

            # If content changed, create a new embedding
            if request.content is not None:
                embedding = model.encode(new_content).tolist()

                cursor.execute(
                    """
                    UPDATE memories
                    SET
                        content = %s,
                        metadata = %s,
                        memory_weight = %s,
                        embedding = %s::vector
                    WHERE id = %s
                    RETURNING id, content, metadata, memory_weight
                    """,
                    (
                        new_content,
                        new_metadata,
                        new_weight,
                        embedding,
                        memory_id,
                    ),
                )

            else:
                cursor.execute(
                    """
                    UPDATE memories
                    SET
                        metadata = %s,
                        memory_weight = %s
                    WHERE id = %s
                    RETURNING id, content, metadata, memory_weight
                    """,
                    (
                        new_metadata,
                        new_weight,
                        memory_id,
                    ),
                )

            row = cursor.fetchone()

        connection.commit()

    finally:
        connection.close()

    if request.content is not None:
        update_memory_graph(
            memory_id=memory_id,
            content=new_content,
        )

    return {
        "id": str(row[0]),
        "content": row[1],
        "metadata": row[2],
        "memory_weight": row[3],
        "message": "Memory updated successfully",
    }
@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM memories
                WHERE id = %s
                RETURNING id, content
                """,
                (memory_id,),
            )

            row = cursor.fetchone()

            if row is None:
                return {
                    "error": "Memory not found"
                }

        connection.commit()

    finally:
        connection.close()

    delete_memory_graph(memory_id)

    return {
        "id": str(row[0]),
        "content": row[1],
        "message": "Memory deleted successfully",
    }
