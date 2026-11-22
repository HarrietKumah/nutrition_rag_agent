# this file is for the retrieval part of RAG -- but just for testing purposes???

from retriever import get_retriever

questions = [
    "what is the sodium limit for CKD patients?",
    "To have blood pressure of less than 140/90 mmHg, what should you limit your sodium intake to?",
]

retriever = get_retriever()

for question in questions:
    print(f"\n=== QUESTION: {question} ===")
    nodes = retriever.retrieve(question)
    for node in nodes:
        print(f"score: {node.score:.3f}")
        print(node.text)
        print("---")

