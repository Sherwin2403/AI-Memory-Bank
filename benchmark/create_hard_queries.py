import json
import os
import re

import psycopg
from dotenv import load_dotenv


load_dotenv()

INPUT_FILE = "benchmark/controlled_dataset.json"
OUTPUT_FILE = "benchmark/hard_dataset.json"

BENCHMARK_NAME = "raw_v1"

USER_ID = "dc2bea0a-3531-46e6-845a-d39d20ae9a00"


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def load_source_records():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_memory_contents():
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

            rows = cursor.fetchall()

        return {
            str(memory_id): content
            for memory_id, content in rows
        }

    finally:
        connection.close()


def extract_focus(content, topic):
    patterns = [
        r"\bfor (.+?)\.$",
        r"\bto (.+?)\.$",
        r"\band (.+?)\.$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            content,
            flags=re.IGNORECASE,
        )

        if match:
            focus = match.group(1).strip()

            if len(focus) >= 5:
                return focus

    return topic


def create_hard_queries():

    source_records = load_source_records()
    memory_contents = load_memory_contents()

    queries = []

    single_memory_records = [
        item
        for item in source_records
        if len(item["relevant_memory_ids"]) == 1
    ]

    # ---------------------------------------------------------
    # Three purpose-based paraphrases per memory
    # ---------------------------------------------------------

    for item in single_memory_records:

        memory_id = item["relevant_memory_ids"][0]
        topic = item["topic"]

        content = memory_contents.get(
            memory_id,
            "",
        )

        focus = extract_focus(
            content,
            topic,
        )

        templates = [
            f"Which technology or concept am I learning for {focus}?",
            f"What am I currently studying to support {focus}?",
            f"Which area of my work is connected to {focus}?",
        ]

        for query in templates:

            queries.append(
                {
                    "query": query,
                    "relevant_memory_ids": [
                        memory_id
                    ],
                    "category": item["category"],
                    "topic": topic,
                    "type": "purpose_paraphrase",
                }
            )

    # ---------------------------------------------------------
    # Multi-memory queries
    # ---------------------------------------------------------

    for i in range(
        0,
        len(single_memory_records) - 1,
        2,
    ):

        first = single_memory_records[i]
        second = single_memory_records[i + 1]

        first_id = first["relevant_memory_ids"][0]
        second_id = second["relevant_memory_ids"][0]

        first_content = memory_contents.get(
            first_id,
            "",
        )

        second_content = memory_contents.get(
            second_id,
            "",
        )

        first_focus = extract_focus(
            first_content,
            first["topic"],
        )

        second_focus = extract_focus(
            second_content,
            second["topic"],
        )

        queries.append(
            {
                "query": (
                    f"Which two areas am I working with "
                    f"that relate to {first_focus} and "
                    f"{second_focus}?"
                ),
                "relevant_memory_ids": [
                    first_id,
                    second_id,
                ],
                "category": first["category"],
                "topic": (
                    f"{first['topic']} + "
                    f"{second['topic']}"
                ),
                "type": "multi_memory",
            }
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            queries,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("HARD BENCHMARK DATASET CREATED")
    print("=" * 70)
    print()
    print(
        f"Source records: "
        f"{len(source_records)}"
    )
    print(
        f"Single-memory records: "
        f"{len(single_memory_records)}"
    )
    print(
        f"Hard queries: "
        f"{len(queries)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print()
    print("=" * 70)


if __name__ == "__main__":
    create_hard_queries()
