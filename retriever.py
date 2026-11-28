# connects to chroma --> wrap in chromavectorstore --> loads index --> creates retriever

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore 

load_dotenv()
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def _get_retriever_for_collection(collection_name, similarity_top_k=3):
  # reconnect to existing chroma collection
  chroma_client = chromadb.PersistentClient(path="./chroma_db")
  chroma_collection = chroma_client.get_collection(collection_name)

  # wrap collection in llama index vector store adapter
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

  # load index from that vector store; no embedding just reading and attaching
  index = VectorStoreIndex.from_vector_store(vector_store)

  # create a retriver form the index
  retriever = index.as_retriever(similarity_top_k=similarity_top_k)
  return retriever

def get_shared_retriever(similarity_top_k=3):
  return _get_retriever_for_collection("health_guidelines", similarity_top_k)

def get_patient_retriever(patient_id, similarity_top_k=3):
  return _get_retriever_for_collection(f"patient_{patient_id}_library", similarity_top_k)