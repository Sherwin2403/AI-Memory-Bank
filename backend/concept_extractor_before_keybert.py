import re


def extract_concepts(text: str) -> list[str]:
    """
    Simple deterministic concept extractor for the initial MVP.

    It looks for known project concepts in the memory text.
    """

    known_concepts = [
        "Python",
        "Artificial Intelligence",
        "Neo4j",
        "PostgreSQL",
        "pgvector",
        "FastAPI",
        "Streamlit",
        "Docker",
    ]

    text_lower = text.lower()

    concepts = []

    for concept in known_concepts:
        if concept.lower() in text_lower:
            concepts.append(concept)

    return concepts
