import logging
import sys
import os

import qdrant_client
#from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext,set_global_service_context

from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.storage.storage_context import StorageContext
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings

# docker run -d --name qdrant -p 6333:6333 qdrant/qdrant


QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "index-resume"
MISTRAL_API_KEY =   "CKM7tMrMkngxZrlKE9cP8Qjk5x6jIVYY"
#DIRECTORY_PATH = "/root/SamIA/resumes_hw2/"
DIRECTORY_PATH = "/home/jminiko/insync/developpement/python/21talents/SamIA/llamaindex/resume_main/"
logging.info("Initializing llm")

llm = MistralAI(api_key=MISTRAL_API_KEY, model="mistral-small")
embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=MISTRAL_API_KEY)

Settings.llm = llm
Settings.embed_model = embed_model
#Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900


logging.info("Initializing vector store...")
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
)
vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

logging.info("Loading documents...")
documents = SimpleDirectoryReader(
    input_dir=DIRECTORY_PATH,
    recursive=True,
).load_data(show_progress=True)
logging.info(f"documents : {len(documents)}")

logging.info("Indexing...")
index = VectorStoreIndex.from_documents(
    documents,storage_context=storage_context,  show_progress=True
)
