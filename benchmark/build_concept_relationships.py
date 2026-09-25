from backend.neo4j_client import driver


def build_relationships():

    with driver.session() as session:

        result = session.run(
            """
            MATCH (m:Memory)-[:ABOUT]->(c:Concept)
            RETURN m.id AS memory_id,
                   collect(c.name) AS concepts
            """
        )

        memory_concepts = [
            (
                record["memory_id"],
                [
                    concept
                    for concept in record["concepts"]
                    if concept
                ],
            )
            for record in result
        ]

        relationship_count = 0

        for memory_id, concepts in memory_concepts:

            for i in range(len(concepts)):

                for j in range(i + 1, len(concepts)):

                    first = concepts[i]
                    second = concepts[j]

                    session.run(
                        """
                        MATCH (c1:Concept {name: $first})
                        MATCH (c2:Concept {name: $second})

                        MERGE (c1)-[:RELATED_TO]->(c2)
                        MERGE (c2)-[:RELATED_TO]->(c1)
                        """,
                        first=first,
                        second=second,
                    )

                    relationship_count += 2

    print()
    print("=" * 70)
    print("CONCEPT RELATIONSHIPS CREATED")
    print("=" * 70)
    print(
        f"Relationships created/merged: "
        f"{relationship_count}"
    )
    print("=" * 70)


if __name__ == "__main__":
    build_relationships()

