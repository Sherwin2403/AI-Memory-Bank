import spacy


nlp = spacy.load("en_core_web_sm")


def normalize_concept(concept: str) -> str:
    """
    Normalize concept names while preserving known technical casing.
    """

    concept = concept.strip()

    special_cases = {
        "python": "Python",
        "neo4j": "Neo4j",
        "postgresql": "PostgreSQL",
        "pgvector": "pgvector",
        "fastapi": "FastAPI",
        "streamlit": "Streamlit",
        "docker": "Docker",
    }

    return special_cases.get(
        concept.lower(),
        concept.title(),
    )


def extract_concepts(text: str) -> list[str]:
    """
    Extract candidate concepts using noun chunks
    and named entities.
    """

    doc = nlp(text)

    concepts = set()

    # Named entities
    for entity in doc.ents:
        concept = entity.text.strip()

        if len(concept) >= 3:
            concepts.add(normalize_concept(concept))

    # Noun chunks
    for chunk in doc.noun_chunks:
        concept = chunk.text.strip()

        if len(concept) < 3:
            continue

        concepts.add(normalize_concept(concept))

    return sorted(concepts)
