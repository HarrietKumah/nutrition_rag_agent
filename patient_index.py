# writes a single patient's uploaded document into their own collection
import os
import chromadb
from llama_index.core import SimpleDirectoryReader

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from dotenv import load_dotenv
load_dotenv()

from llama_index.embeddings.openai import OpenAIEmbedding

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")




def add_patient_document(patient_id, file_path, doc_type, date=None):
    def file_metadata_fn(path):
        metadata = {"doc_type": doc_type}
        if date:
            metadata["date"] = date
        return metadata

    reader = SimpleDirectoryReader(input_files=[file_path], file_metadata=file_metadata_fn)
    documents = reader.load_data()

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection(f"patient_{patient_id}_library")

    # stores vectors
    # wraps the chroma collection in an interface that llama index can read
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # tells the index to store vectors in Chroma specifically; makes it persistent in a databse
    # keeps other storage (docs, index structure) as in-memory defaults
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # llama index splits each document into chunks then calls the...
    # .. embedding model (openai_api_key) to embed each chunk's text ...
    # ..and writes (llama index) each chunk+embedding into the chroma vector store
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return {
        "patient_id": patient_id,
        "doc_type": doc_type,
        "date": date,
        "pages_indexed": len(documents),
        "total_chunks_in_library": chroma_collection.count(),
    }


if __name__ == "__main__":
    print(add_patient_document(
        patient_id="jane_demo",
        file_path="test_data/fake_ehr.txt",
        doc_type="ehr",
    ))
    print(add_patient_document(
        patient_id="jane_demo",
        file_path="test_data/fake_labs_jan.txt",
        doc_type="lab_result",
        date="2026-01-15",
    ))