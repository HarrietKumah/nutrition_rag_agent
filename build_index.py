### llama index Loads the health guideline pdfs into the reader then ...
# ... splits each document into chunks then embeds the chunks using the ...
# ... openai embed model and stores them in a local chroma database


import os
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

from llama_index.embeddings.openai import OpenAIEmbedding

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


reader = SimpleDirectoryReader(input_dir="docs") # input the documents into the reader
# reads file and returns a list of doucment objects (one per page of documents); each document obj is ...
# ...raw extracted text no chunking or embedding yet
documents = reader.load_data() 

# creates a reopens chroma database; will create chroma_db folder if it doesn't exist
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collections are like tables in a database; each "row" is one chunk of text
chroma_collection = chroma_client.get_or_create_collection("health_guidelines")

# stores vectors
# wraps the chroma collection in an interface that llama index can read
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# tells the index to store vectors in Chroma specifically; makes it persistent in a databse
# keeps other storage (docs, index structure) as in-memory defaults
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# llama index splits each document into chunks then calls the...
# .. embedding model (openai_api_key) to embed each chunk's text ...
# ..and writes (llama index) each chunk+embedding into the chroma vector store
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


print(f"Indexed {len(documents)} pages into {chroma_collection.count()} chunks.")

print(documents[0].text[:300])     # raw extracted text from page 1, first 300 chars
print(documents[0].metadata)       # e.g. {'page_label': '1', 'file_name': 'ckd-diet-...pdf', ...}

sample = chroma_collection.peek(limit=1) # grabs a small sample of rows from the collection to see what it looks like
print(sample["documents"])   # the actual chunk text stored
print(sample["metadatas"])   # metadata for that chunk
print(len(sample["embeddings"][0]))   # dim of the embedding vector — should be 1536 for openai
