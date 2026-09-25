from backend.concept_extractor import extract_concepts


texts = [
    "I am learning Python, artificial intelligence, and Neo4j.",
    "I enjoy building distributed systems and studying computer networks.",
    "I want to learn more about machine learning and natural language processing.",
]


for text in texts:
    print("\nText:")
    print(text)

    concepts = extract_concepts(text)

    print("Extracted concepts:")

    for concept in concepts:
        print("-", concept)
