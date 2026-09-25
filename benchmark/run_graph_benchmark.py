import json
import os
import statistics
import time

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.concept_extractor import extract_concepts
from backend.neo4j_client import driver


load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

USER_ID = "dc2bea0a-3531-46e6-845a-d39d20ae9a00"

BENCHMARK_NAME = "raw_v1"

TOP_K = 5
HYBRID_CANDIDATES = 20

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
    with open(
        "benchmark/graph_dataset.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def retrieve_vector_only(
    query_embedding,
    user_id,
    limit,
):
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
                  AND metadata->>'benchmark' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    user_id,
                    BENCHMARK_NAME,
                    query_embedding,
                    limit,
                ),
            )

            return cursor.fetchall()

    finally:
        connection.close()


def retrieve_vector_lifecycle(
    query_embedding,
    user_id,
    limit,
):
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
                  AND metadata->>'benchmark' = %s
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
                    BENCHMARK_NAME,
                    query_embedding,
                    limit,
                ),
            )

            return cursor.fetchall()

    finally:
        connection.close()


def get_graph_scores(
    query_concepts,
    memory_ids,
):
    scores = {}

    if not memory_ids:
        return scores

    with driver.session() as session:

        result = session.run(
            """
            MATCH (m:Memory)-[:ABOUT]->(c:Concept)
            WHERE m.id IN $memory_ids
            RETURN
                m.id AS memory_id,
                collect(toLower(c.name)) AS concepts
            """,
            memory_ids=memory_ids,
        )

        for record in result:

            memory_id = str(
                record["memory_id"]
            )

            memory_concepts = {
                concept.strip().lower()
                for concept in record["concepts"]
            }

            query_set = {
                concept.strip().lower()
                for concept in query_concepts
            }

            if not query_set:
                scores[memory_id] = 0.0
                continue

            overlap = (
                memory_concepts
                & query_set
            )

            scores[memory_id] = (
                len(overlap)
                / len(query_set)
            )

    for memory_id in memory_ids:

        if memory_id not in scores:
            scores[memory_id] = 0.0

    return scores
def retrieve_hybrid(
    query_embedding,
    query_concepts,
    user_id,
    limit,
):
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
                  AND metadata->>'benchmark' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    user_id,
                    BENCHMARK_NAME,
                    query_embedding,
                    HYBRID_CANDIDATES,
                ),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    memory_ids = [
        str(row[0])
        for row in rows
    ]

    graph_scores = get_graph_scores(
        query_concepts,
        memory_ids,
    )

    scored = []

    for row in rows:

        memory_id = str(row[0])
        content = row[1]
        memory_weight = float(row[2])
        access_count = int(row[3])
        distance = float(row[4])

        similarity = 1.0 - distance

        access_frequency = (
            min(access_count, 10)
            / 10.0
        )

        graph_relevance = graph_scores.get(
            memory_id,
            0.0,
        )

        score = (
            similarity * 0.6
            + memory_weight * 0.2
            + access_frequency * 0.1
            + graph_relevance * 0.1
        )

        scored.append(
            (
                memory_id,
                content,
                memory_weight,
                access_count,
                distance,
                graph_relevance,
                score,
            )
        )

    scored.sort(
        key=lambda row: row[-1],
        reverse=True,
    )

    return scored[:limit]

def recall_at_k(
    results,
    relevant_ids,
    k,
):
    returned_ids = {
        str(row[0])
        for row in results[:k]
    }

    relevant_ids = {
        str(memory_id)
        for memory_id in relevant_ids
    }

    if not relevant_ids:
        return 0.0

    return (
        len(returned_ids & relevant_ids)
        / len(relevant_ids)
    )


def percentile(
    values,
    percentile_value,
):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = (
        (len(sorted_values) - 1)
        * percentile_value
        / 100
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(sorted_values) - 1,
    )

    weight = index - lower

    return (
        sorted_values[lower]
        + (
            sorted_values[upper]
            - sorted_values[lower]
        )
        * weight
    )


def run_benchmark():

    dataset = load_dataset()

    vector_recalls_1 = []
    vector_recalls_5 = []

    lifecycle_recalls_1 = []
    lifecycle_recalls_5 = []

    hybrid_recalls_1 = []
    hybrid_recalls_5 = []

    vector_latencies = []
    lifecycle_latencies = []
    hybrid_latencies = []

    print()
    print("=" * 70)
    print("HYBRID AI MEMORY BANK BENCHMARK")
    print("=" * 70)
    print()

    print(
        f"Queries: {len(dataset)}"
    )

    print(
        f"Embedding model: {MODEL_NAME}"
    )

    print(
        f"Benchmark: {BENCHMARK_NAME}"
    )

    print()

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        query = item["query"]
        relevant_ids = item[
            "relevant_memory_ids"
        ]

        query_embedding = model.encode(
            query
        ).tolist()

        query_concepts = extract_concepts(
            query
        )

        # -----------------------------------------------------
        # Vector
        # -----------------------------------------------------

        start = time.perf_counter()

        vector_results = retrieve_vector_only(
            query_embedding,
            USER_ID,
            TOP_K,
        )

        end = time.perf_counter()

        vector_latency = (
            end - start
        ) * 1000

        vector_latencies.append(
            vector_latency
        )

        vector_recalls_1.append(
            recall_at_k(
                vector_results,
                relevant_ids,
                1,
            )
        )

        vector_recalls_5.append(
            recall_at_k(
                vector_results,
                relevant_ids,
                5,
            )
        )

        # -----------------------------------------------------
        # Lifecycle
        # -----------------------------------------------------

        start = time.perf_counter()

        lifecycle_results = (
            retrieve_vector_lifecycle(
                query_embedding,
                USER_ID,
                TOP_K,
            )
        )

        end = time.perf_counter()

        lifecycle_latency = (
            end - start
        ) * 1000

        lifecycle_latencies.append(
            lifecycle_latency
        )

        lifecycle_recalls_1.append(
            recall_at_k(
                lifecycle_results,
                relevant_ids,
                1,
            )
        )

        lifecycle_recalls_5.append(
            recall_at_k(
                lifecycle_results,
                relevant_ids,
                5,
            )
        )

        # -----------------------------------------------------
        # Hybrid
        # -----------------------------------------------------

        start = time.perf_counter()

        hybrid_results = retrieve_hybrid(
            query_embedding,
            query_concepts,
            USER_ID,
            TOP_K,
        )

        end = time.perf_counter()

        hybrid_latency = (
            end - start
        ) * 1000

        hybrid_latencies.append(
            hybrid_latency
        )

        hybrid_recalls_1.append(
            recall_at_k(
                hybrid_results,
                relevant_ids,
                1,
            )
        )

        hybrid_recalls_5.append(
            recall_at_k(
                hybrid_results,
                relevant_ids,
                5,
            )
        )

        if index % 25 == 0:

            print(
                f"[{index}/{len(dataset)}] processed"
            )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
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

    print("HYBRID VECTOR + GRAPH")

    print(
        f"Recall@1       : "
        f"{statistics.mean(hybrid_recalls_1):.4f}"
    )

    print(
        f"Recall@5       : "
        f"{statistics.mean(hybrid_recalls_5):.4f}"
    )

    print(
        f"Mean latency   : "
        f"{statistics.mean(hybrid_latencies):.2f} ms"
    )

    print(
        f"Median latency : "
        f"{statistics.median(hybrid_latencies):.2f} ms"
    )

    print(
        f"P95 latency    : "
        f"{percentile(hybrid_latencies, 95):.2f} ms"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
