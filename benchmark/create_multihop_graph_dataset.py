import json

from backend.neo4j_client import driver


OUTPUT_FILE = "benchmark/multihop_graph_dataset.json"


CONCEPTS = [
    "Python",
    "PostgreSQL",
    "Neo4j",
    "Machine Learning",
    "Natural Language Processing",
    "FastAPI",
    "Docker",
    "Java",
    "TCP",
    "K-Means",
]


def create_dataset():

    dataset = []

    with driver.session() as session:

        for concept in CONCEPTS:

            result = session.run(
                """
                MATCH (q:Concept)
                WHERE toLower(q.name) = toLower($concept)

                OPTIONAL MATCH
                    (direct:Memory)-[:ABOUT]->(q)

                OPTIONAL MATCH
                    (q)<-[:ABOUT]-(source:Memory)
                    -[:ABOUT]->(shared:Concept)
                    <-[:ABOUT]-(related:Memory)

                WITH
                    collect(DISTINCT direct.id) AS direct_ids,
                    collect(DISTINCT related.id) AS related_ids

                RETURN direct_ids, related_ids
                """,
                concept=concept,
            )

            record = result.single()

            if record is None:
                continue

            direct_ids = {
                str(memory_id)
                for memory_id in record["direct_ids"]
                if memory_id is not None
            }

            indirect_ids = {
                str(memory_id)
                for memory_id in record["related_ids"]
                if memory_id is not None
            }

            # Do not count the directly connected memories twice.
            indirect_ids -= direct_ids

            all_relevant_ids = (
                direct_ids
                | indirect_ids
            )

            if not all_relevant_ids:
                print(
                    f"WARNING: No graph connections for {concept}"
                )
                continue

            dataset.append(
                {
                    "query": (
                        f"Which memories are connected to "
                        f"{concept}, either directly or "
                        f"through another shared concept?"
                    ),
                    "relevant_memory_ids": sorted(
                        all_relevant_ids
                    ),
                    "direct_memory_ids": sorted(
                        direct_ids
                    ),
                    "indirect_memory_ids": sorted(
                        indirect_ids
                    ),
                    "concept": concept,
                    "type": "multi_hop_graph",
                }
            )

            print(
                f"{concept}: "
                f"{len(direct_ids)} direct, "
                f"{len(indirect_ids)} indirect, "
                f"{len(all_relevant_ids)} total"
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
    print("MULTI-HOP GRAPH DATASET CREATED")
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



