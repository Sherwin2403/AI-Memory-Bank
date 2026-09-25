from backend.neo4j_client import create_concept_graph


create_concept_graph(
    memory_id="640af191-bf07-4dbb-8dba-f936f0da6869",
    concepts=[
        "Python",
        "Artificial Intelligence",
        "Neo4j",
    ],
)

print("Concept graph created successfully!")
