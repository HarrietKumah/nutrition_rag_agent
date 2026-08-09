import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")



# 2. reconnect to exisitng chroma collectiom
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("health_guidelines")

# 3. wrap collection in llama index vector store adapter
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 4. load index from that vector store; no embedding just reading and attaching
index = VectorStoreIndex.from_vector_store(vector_store)

# 5. create a retriver form the index
retriever = index.as_retriever(similarity_top_k=3)


questions = [
    "what is the sodium limit for CKD patients?",
    "To have blood pressure of less than 140/90 mmHg, what should you limit your sodium intake to?",
]

for question in questions:
    print(f"\n=== QUESTION: {question} ===")
    nodes = retriever.retrieve(question)
    for node in nodes:
        print(f"score: {node.score:.3f}")
        print(node.text)
        print("---")

