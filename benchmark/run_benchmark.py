import json
import os
import statistics
import time

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def load_dataset():
    with open("benchmark/dataset.json", "r", encoding="utf-8") as file:
        return json.load(file)


def get_user_id():
    return "dc2bea0a-3531-46e6-845a-d39d20ae9a00"


def retrieve_vector_only(query_embedding, user_id, limit):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    embedding <=> %s::vector AS distance
                FROM memories
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    user_id,
                    query_embedding,
                    limit,
                ),
            )

            rows = cursor.fetchall()

        return rows

    finally:
        connection.close()


def retrieve_vector_lifecycle(query_embedding, user_id, limit):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    memory_weight,
                    access_count,
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
                    user_id,
                    query_embedding,
                    limit,
                ),
            )

            rows = cursor.fetchall()

        return rows

    finally:
        connection.close()


def recall_at_k(results, relevant_ids, k):
    returned_ids = {str(row[0]) for row in results[:k]}
    relevant_ids = {str(memory_id) for memory_id in relevant_ids}

    if not relevant_ids:
        return 0.0

    return len(returned_ids & relevant_ids) / len(relevant_ids)


def percentile(values, percentile):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = (len(sorted_values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = index - lower

    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * weight
    )


def run_benchmark():
    dataset = load_dataset()
    user_id = get_user_id()

    vector_recalls_1 = []
    vector_recalls_5 = []

    lifecycle_recalls_1 = []
    lifecycle_recalls_5 = []

    vector_latencies = []
    lifecycle_latencies = []

    print()
    print("=" * 70)
    print("AI MEMORY BANK - RETRIEVAL BENCHMARK")
    print("=" * 70)
    print()

    print(f"Queries: {len(dataset)}")
    print(f"Embedding model: {MODEL_NAME}")
    print()

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("Running warm-up...")

    warmup_embedding = model.encode(
        "What programming language am I learning?"
    ).tolist()

    retrieve_vector_only(
        warmup_embedding,
        user_id,
        5,
    )

    retrieve_vector_lifecycle(
        warmup_embedding,
        user_id,
        5,
    )

    print("Warm-up complete.")
    print()

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    for index, item in enumerate(dataset, start=1):

        query = item["query"]
        relevant_ids = item["relevant_memory_ids"]

        print(f"[{index}/{len(dataset)}] {query}")

        # -----------------------------------------------------
        # Generate embedding once
        # -----------------------------------------------------

        query_embedding = model.encode(query).tolist()

        # -----------------------------------------------------
        # Vector-only
        # -----------------------------------------------------

        start = time.perf_counter()

        vector_results = retrieve_vector_only(
            query_embedding,
            user_id,
            5,
        )

        end = time.perf_counter()

        vector_latency_ms = (end - start) * 1000

        vector_latencies.append(vector_latency_ms)

        vector_recalls_1.append(
            recall_at_k(vector_results, relevant_ids, 1)
        )

        vector_recalls_5.append(
            recall_at_k(vector_results, relevant_ids, 5)
        )

        # -----------------------------------------------------
        # Vector + lifecycle
        # -----------------------------------------------------

        start = time.perf_counter()

        lifecycle_results = retrieve_vector_lifecycle(
            query_embedding,
            user_id,
            5,
        )

        end = time.perf_counter()

        lifecycle_latency_ms = (end - start) * 1000

        lifecycle_latencies.append(lifecycle_latency_ms)

        lifecycle_recalls_1.append(
            recall_at_k(lifecycle_results, relevant_ids, 1)
        )

        lifecycle_recalls_5.append(
            recall_at_k(lifecycle_results, relevant_ids, 5)
        )


        print(
            f"    Vector-only   : "
            f"Recall@1={vector_recalls_1[-1]:.2f}, "
            f"Recall@5={vector_recalls_5[-1]:.2f}, "
            f"Latency={vector_latency_ms:.2f} ms"
        )

        print("    Vector-only top results:")

        for rank, row in enumerate(vector_results, start=1):
            memory_id = str(row[0])
            content = row[1]
            distance = float(row[2])

            print(
                f"        {rank}. "
                f"{memory_id[:8]}... | "
                f"distance={distance:.4f} | "
                f"{content}"
            )

        print(
            f"    Vector+lifecycle: "
            f"Recall@1={lifecycle_recalls_1[-1]:.2f}, "
            f"Recall@5={lifecycle_recalls_5[-1]:.2f}, "
            f"Latency={lifecycle_latency_ms:.2f} ms"
        )

        print("    Vector+lifecycle top results:")

        for rank, row in enumerate(lifecycle_results, start=1):
            memory_id = str(row[0])
            content = row[1]
            memory_weight = float(row[2])
            access_count = int(row[3])
            distance = float(row[4])

            similarity = 1 - distance
            access_frequency = min(access_count, 10) / 10.0

            score = (
                similarity * 0.7
                + memory_weight * 0.2
                + access_frequency * 0.1
            )

            print(
                f"        {rank}. "
                f"{memory_id[:8]}... | "
                f"similarity={similarity:.4f} | "
                f"weight={memory_weight:.2f} | "
                f"access={access_count} | "
                f"score={score:.4f} | "
                f"{content}"
            )

        print()
    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()

    print("VECTOR-ONLY")
    print(
        f"Recall@1       : "
        f"{statistics.mean(vector_recalls_1):.4f}"
    )
    print(
        f"Recall@5       : "
        f"{statistics.mean(vector_recalls_5):.4f}"
    )
    print(
        f"Mean latency   : "
        f"{statistics.mean(vector_latencies):.2f} ms"
    )
    print(
        f"Median latency : "
        f"{statistics.median(vector_latencies):.2f} ms"
    )
    print(
        f"P95 latency    : "
        f"{percentile(vector_latencies, 95):.2f} ms"
    )

    print()

    print("VECTOR + LIFECYCLE")
    print(
        f"Recall@1       : "
        f"{statistics.mean(lifecycle_recalls_1):.4f}"
    )
    print(
        f"Recall@5       : "
        f"{statistics.mean(lifecycle_recalls_5):.4f}"
    )
    print(
        f"Mean latency   : "
        f"{statistics.mean(lifecycle_latencies):.2f} ms"
    )
    print(
        f"Median latency : "
        f"{statistics.median(lifecycle_latencies):.2f} ms"
    )
    print(
        f"P95 latency    : "
        f"{percentile(lifecycle_latencies, 95):.2f} ms"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
