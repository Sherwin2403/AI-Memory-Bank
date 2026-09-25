import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
)

with driver.session() as session:
    result = session.run(
        "RETURN 'Neo4j connection works!' AS status"
    )
    print(result.single()["status"])

driver.close()
