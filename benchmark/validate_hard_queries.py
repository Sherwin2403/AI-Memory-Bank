import json
import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

INPUT_FILE = "benchmark/hard_dataset.json"
OUTPUT_FILE = "benchmark/validated_dataset.json"

USER_ID = "dc2bea0a-3531-46e6-845a-d39d20ae9a00"
BENCHMARK_NAME = "raw_v1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_MARGIN = 0.08


model = SentenceTransformer(MODEL_NAME)


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def load_queries():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_memories():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content
                FROM memories
                WHERE user_id = %s
                  AND metadata->>'benchmark' = %s
                """,
                (
                    USER_ID,
                    BENCHMARK_NAME,
                ),
            )

            return cursor.fetchall()

    finally:
        connection.close()


def get_similarities(query_embedding, memories):

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            similarities = []

            for memory_id, content in memories:

                cursor.execute(
                    """
                    SELECT
                        1 - (
                            embedding <=> %s::vector
                        ) AS similarity
                    FROM memories
                    WHERE id = %s
                    """,
                    (
                        query_embedding,
                        memory_id,
                    ),
                )

                row = cursor.fetchone()

                if row is not None:

                    similarities.append(
                        (
                            str(memory_id),
                            float(row[0]),
                            content,
                        )
                    )

        similarities.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return similarities

    finally:
        connection.close()


def main():

    queries = load_queries()
    memories = get_memories()

    validated = []
    rejected = []

    print()
    print("=" * 70)
    print("VALIDATING HARD BENCHMARK")
    print("=" * 70)
    print()

    print(
        f"Input queries: {len(queries)}"
    )

    print(
        f"Benchmark memories: {len(memories)}"
    )

    print(
        f"Minimum similarity margin: {MIN_MARGIN}"
    )

    print()

    for index, item in enumerate(
        queries,
        start=1,
    ):

        query = item["query"]

        relevant_ids = {
            str(memory_id)
            for memory_id
            in item["relevant_memory_ids"]
        }

        query_embedding = model.encode(
            query
        ).tolist()

        similarities = get_similarities(
            query_embedding,
            memories,
        )

        if not similarities:
            rejected.append(item)
            continue

        relevant_scores = [
            similarity
            for memory_id, similarity, _
            in similarities
            if memory_id in relevant_ids
        ]

        if not relevant_scores:
            rejected.append(item)
            continue

        best_relevant_score = max(
            relevant_scores
        )

        competing_scores = [
            similarity
            for memory_id, similarity, _
            in similarities
            if memory_id not in relevant_ids
        ]

        if competing_scores:

            best_competing_score = max(
                competing_scores
            )

            margin = (
                best_relevant_score
                - best_competing_score
            )

        else:
            margin = 1.0

        if margin >= MIN_MARGIN:

            validated.append(item)

        else:

            rejected.append(item)

        if index % 25 == 0:

            print(
                f"[{index}/{len(queries)}] "
                f"validated={len(validated)} "
                f"rejected={len(rejected)}"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            validated,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Input queries : {len(queries)}"
    )

    print(
        f"Validated     : {len(validated)}"
    )

    print(
        f"Rejected      : {len(rejected)}"
    )

    print(
        f"Output        : {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
