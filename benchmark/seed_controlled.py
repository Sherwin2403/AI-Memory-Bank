import json
import os
import uuid

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.concept_extractor import extract_concepts
from backend.neo4j_client import (
    create_memory_graph,
    create_concept_graph,
)


load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

USER_ID = "dc2bea0a-3531-46e6-845a-d39d20ae9a00"

BENCHMARK_NAME = "raw_v1"


model = SentenceTransformer(MODEL_NAME)


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


memory_groups = {
    "programming": [
        ("Python", "I am learning Python for general programming."),
        ("Java", "I am studying Java for object-oriented programming."),
        ("Rust", "I am exploring Rust for systems programming."),
        ("C++", "I am practicing C++ for algorithmic programming."),
        ("Go", "I am learning Go for concurrent applications."),
        ("JavaScript", "I am studying JavaScript for web development."),
        ("TypeScript", "I am learning TypeScript for type-safe web applications."),
        ("C", "I am practicing C for low-level programming."),
        ("Kotlin", "I am studying Kotlin for application development."),
        ("Swift", "I am learning Swift for application programming."),
    ],
    "databases": [
        ("PostgreSQL", "I am learning PostgreSQL for relational database management."),
        ("MySQL", "I am studying MySQL for relational database applications."),
        ("MongoDB", "I am exploring MongoDB for document-oriented data storage."),
        ("Neo4j", "I am learning Neo4j for graph-based data management."),
        ("Redis", "I am studying Redis for in-memory data storage."),
        ("SQLite", "I am learning SQLite for lightweight relational databases."),
        ("Cassandra", "I am exploring Cassandra for distributed data storage."),
        ("Oracle", "I am studying Oracle Database for enterprise data management."),
        ("MariaDB", "I am learning MariaDB for relational database applications."),
        ("Elasticsearch", "I am exploring Elasticsearch for search-oriented data systems."),
    ],
    "artificial_intelligence": [
        ("Machine Learning", "I am studying machine learning for predictive systems."),
        ("Natural Language Processing", "I am learning natural language processing for text analysis."),
        ("Computer Vision", "I am studying computer vision for image understanding."),
        ("Deep Learning", "I am exploring deep learning for complex pattern recognition."),
        ("Reinforcement Learning", "I am learning reinforcement learning for sequential decision making."),
        ("Transformers", "I am studying transformer architectures for modern AI systems."),
        ("Generative AI", "I am learning generative AI for content generation."),
        ("Retrieval Augmented Generation", "I am studying retrieval augmented generation for knowledge-aware AI."),
        ("Large Language Models", "I am exploring large language models for language-based applications."),
        ("Embeddings", "I am learning embeddings for semantic representation of information."),
    ],
    "networking": [
        ("TCP", "I am studying TCP for reliable network communication."),
        ("UDP", "I am learning UDP for connectionless network communication."),
        ("DNS", "I am studying DNS for domain name resolution."),
        ("DHCP", "I am learning DHCP for automatic network configuration."),
        ("ARP", "I am studying ARP for mapping IP addresses to hardware addresses."),
        ("ICMP", "I am learning ICMP for network diagnostics and control messages."),
        ("IP fragmentation", "I am studying IP fragmentation and packet reassembly."),
        ("Routing", "I am learning routing for forwarding packets between networks."),
        ("Subnetting", "I am studying subnetting for organizing IP networks."),
        ("HTTP", "I am learning HTTP for communication between web clients and servers."),
    ],
    "backend": [
        ("FastAPI", "I am using FastAPI for building Python web APIs."),
        ("Flask", "I am studying Flask for lightweight Python web applications."),
        ("Django", "I am learning Django for full-stack Python web development."),
        ("Node.js", "I am exploring Node.js for server-side JavaScript applications."),
        ("REST API", "I am studying REST APIs for service-to-service communication."),
        ("Docker", "I am learning Docker for containerized application development."),
        ("Uvicorn", "I am using Uvicorn to serve Python web applications."),
        ("Pydantic", "I am studying Pydantic for data validation in Python applications."),
        ("JWT", "I am learning JSON Web Tokens for API authentication."),
        ("Microservices", "I am studying microservices for distributed backend architectures."),
    ],
    "cloud_devops": [
        ("AWS", "I am learning AWS for cloud computing."),
        ("Azure", "I am studying Azure for cloud services."),
        ("Google Cloud", "I am exploring Google Cloud for cloud infrastructure."),
        ("Kubernetes", "I am learning Kubernetes for container orchestration."),
        ("Terraform", "I am studying Terraform for infrastructure as code."),
        ("CI/CD", "I am learning CI/CD for automated software delivery."),
        ("GitHub Actions", "I am exploring GitHub Actions for workflow automation."),
        ("Linux", "I am studying Linux for server and development environments."),
        ("Nginx", "I am learning Nginx for web serving and reverse proxying."),
        ("Monitoring", "I am studying application monitoring for reliable services."),
    ],
    "systems": [
        ("Operating Systems", "I am studying operating systems and system resource management."),
        ("Processes", "I am learning about processes in operating systems."),
        ("Threads", "I am studying threads and concurrent execution."),
        ("Virtual Memory", "I am learning about virtual memory and memory management."),
        ("File Systems", "I am studying file systems and persistent storage."),
        ("Scheduling", "I am learning CPU scheduling algorithms."),
        ("Deadlocks", "I am studying deadlocks and synchronization."),
        ("Distributed Systems", "I am learning distributed systems and coordination."),
        ("Concurrency", "I am studying concurrency in software systems."),
        ("Computer Architecture", "I am exploring computer architecture and processor design."),
    ],
    "data_science": [
        ("Data Mining", "I am studying data mining for extracting useful patterns."),
        ("PCA", "I am learning principal component analysis for dimensionality reduction."),
        ("Clustering", "I am studying clustering algorithms for grouping similar data."),
        ("K-Means", "I am learning K-Means clustering."),
        ("Regression", "I am studying regression for predictive modeling."),
        ("Classification", "I am learning classification techniques for supervised learning."),
        ("Decision Trees", "I am studying decision trees for classification and prediction."),
        ("Random Forest", "I am learning random forests for ensemble learning."),
        ("Feature Engineering", "I am studying feature engineering for machine learning models."),
        ("Data Visualization", "I am learning data visualization for exploratory analysis."),
    ],
    "dbms": [
        ("Normalization", "I am studying database normalization to reduce redundancy."),
        ("Transactions", "I am learning database transactions and consistency."),
        ("ACID", "I am studying ACID properties in database systems."),
        ("Indexes", "I am learning database indexing for faster queries."),
        ("B-Trees", "I am studying B-Tree indexes in database systems."),
        ("HNSW", "I am learning HNSW for efficient vector similarity search."),
        ("SQL joins", "I am studying SQL joins for combining relational data."),
        ("Foreign Keys", "I am learning foreign keys for maintaining database relationships."),
        ("Query Optimization", "I am studying query optimization for efficient database execution."),
        ("Replication", "I am learning database replication for availability and scalability."),
    ],
    "software_engineering": [
        ("Git", "I am learning Git for version control."),
        ("GitHub", "I am using GitHub for collaborative software development."),
        ("Unit Testing", "I am studying unit testing for software quality."),
        ("Debugging", "I am practicing debugging techniques."),
        ("Agile", "I am studying Agile development practices."),
        ("Design Patterns", "I am learning software design patterns."),
        ("Clean Code", "I am studying clean code principles."),
        ("Code Review", "I am learning code review practices."),
        ("Documentation", "I am studying technical documentation practices."),
        ("Software Architecture", "I am learning software architecture and system design."),
    ],
}


def create_memories():
    connection = get_connection()

    all_memories = []

    try:
        with connection.cursor() as cursor:

            # Remove previous version of this benchmark.
            cursor.execute(
                """
                DELETE FROM memories
                WHERE user_id = %s
                  AND metadata->>'benchmark' = %s
                """,
                (USER_ID, BENCHMARK_NAME),
            )

            for category, entries in memory_groups.items():

                for topic, content in entries:

                    memory_id = str(uuid.uuid4())

                    embedding = model.encode(
                        content
                    ).tolist()

                    metadata = json.dumps(
                        {
                            "benchmark": BENCHMARK_NAME,
                            "category": category,
                            "topic": topic,
                        }
                    )

                    cursor.execute(
                        """
                        INSERT INTO memories (
                            id,
                            user_id,
                            content,
                            metadata,
                            embedding,
                            memory_weight,
                            access_count
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s::vector,
                            1.0,
                            0
                        )
                        """,
                        (
                            memory_id,
                            USER_ID,
                            content,
                            metadata,
                            embedding,
                        ),
                    )

                    all_memories.append(
                        {
                            "id": memory_id,
                            "category": category,
                            "topic": topic,
                            "content": content,
                        }
                    )

        connection.commit()

    finally:
        connection.close()

    return all_memories


def create_graph(all_memories):

    for memory in all_memories:

        create_memory_graph(
            user_id=USER_ID,
            memory_id=memory["id"],
            content=memory["content"],
        )

        concepts = extract_concepts(
            memory["content"]
        )

        create_concept_graph(
            memory_id=memory["id"],
            concepts=concepts,
        )


def create_queries(all_memories):

    queries = []

    for memory in all_memories:

        query = (
            f"What am I studying or working with "
            f"related to {memory['topic']}?"
        )

        queries.append(
            {
                "query": query,
                "relevant_memory_ids": [
                    memory["id"]
                ],
                "category": memory["category"],
                "topic": memory["topic"],
            }
        )

    return queries


def main():

    print()
    print("=" * 70)
    print("CREATING CONTROLLED BENCHMARK DATASET")
    print("=" * 70)
    print()

    memories = create_memories()

    print(
        f"Created {len(memories)} benchmark memories."
    )

    print("Creating Neo4j graph data...")

    create_graph(memories)

    print("Graph data created.")

    queries = create_queries(memories)

    with open(
        "benchmark/controlled_dataset.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            queries,
            file,
            indent=2,
        )

    print(
        f"Created {len(queries)} benchmark queries."
    )

    print()
    print(
        "Benchmark version:",
        BENCHMARK_NAME,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
