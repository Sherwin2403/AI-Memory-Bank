from backend.neo4j_client import driver
import json


OUTPUT_FILE = "benchmark/graph_dataset.json"


QUERIES = [
    ("Python", "Which stored memories are connected to Python?"),
    ("PostgreSQL", "What memories are associated with PostgreSQL?"),
    ("Neo4j", "Which memories are related to Neo4j?"),
    ("Machine Learning", "Which memories are connected to machine learning?"),
    (
        "Natural Language Processing",
        "What information is associated with natural language processing?",
    ),
    ("FastAPI", "Which memories are connected to FastAPI?"),
    ("Docker", "What stored memories are related to Docker?"),
    ("Java", "Which memories are associated with Java?"),
    ("TCP", "What memories are connected to TCP?"),
    ("K-Means", "Which stored memories are related to K-Means?"),
]


def create_dataset():
    dataset = []

    with driver.session() as session:

        for concept, query in QUERIES:

            result = session.run(
                """
                MATCH (m:Memory)-[:ABOUT]->(c:Concept)
                WHERE toLower(c.name) = toLower($concept)
                RETURN m.id AS memory_id
                ORDER BY m.id
                """,
                concept=concept,
            )

            memory_ids = [
                str(record["memory_id"])
                for record in result
            ]

            if not memory_ids:
                print(
                    f"WARNING: No memories found for {concept}"
                )
                continue

            dataset.append(
                {
                    "query": query,
                    "relevant_memory_ids": memory_ids,
                    "concept": concept,
                    "type": "graph_concept",
                }
            )

            print(
                f"{concept}: "
                f"{len(memory_ids)} relevant memories"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("GRAPH DATASET CREATED")
    print("=" * 70)
    print(
        f"Queries: {len(dataset)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    create_dataset()
