import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


def create_memory_graph(
    user_id: str,
    memory_id: str,
    content: str,
):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (m:Memory {id: $memory_id})
            SET m.content = $content
            MERGE (u)-[:HAS_MEMORY]->(m)
            """,
            user_id=user_id,
            memory_id=memory_id,
            content=content,
        )

def update_memory_graph(
    memory_id: str,
    content: str,
    concepts: list[str],
):
    with driver.session() as session:
        session.run(
            """
            MATCH (m:Memory {id: $memory_id})
            SET m.content = $content
            WITH m
            OPTIONAL MATCH (m)-[r:ABOUT]->()
            DELETE r
            """,
            memory_id=memory_id,
            content=content,
        )

        for concept in concepts:
            session.run(
                """
                MATCH (m:Memory {id: $memory_id})
                MERGE (c:Concept {name: $concept})
                MERGE (m)-[:ABOUT]->(c)
                """,
                memory_id=memory_id,
                concept=concept,
            )


def delete_memory_graph(memory_id: str):
    with driver.session() as session:
        session.run(
            """
            MATCH (m:Memory {id: $memory_id})
            DETACH DELETE m
            """,
            memory_id=memory_id,
        )

def create_concept_graph(
    memory_id: str,
    concepts: list[str],
):
    with driver.session() as session:
        for concept in concepts:
            session.run(
                """
                MATCH (m:Memory {id: $memory_id})
                MERGE (c:Concept {name: $concept})
                MERGE (m)-[:ABOUT]->(c)
                """,
                memory_id=memory_id,
                concept=concept,
            )

def close_driver():
    driver.close()
