from dotenv import load_dotenv
import os
import openai
import langchain

import qdrant_client
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams,Distance
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai.chat_models import ChatMistralAI
from llama_index.core import Settings
load_dotenv()


#Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name = os.getenv('COLLECTION_NAME_FINAL')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH2"]

client = qdrant_client.QdrantClient(
    url=qdrant_url,
)
print("DBG0100")
if not client.collection_exists(collection_name):
    client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  
    )
print("DBG0200")
def read_pdf(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
print("DBG0300")
print(directory_path)
doc = read_pdf(directory_path)
print("DBG0400")
print(len(doc))

def chunk_data(docs, chunk_size = 1500, chunk_overlap = 10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap= chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc
print("DBG0500")
documents = chunk_data(docs = doc)
print("DBG0500")

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
Settings.embed_model = embeddings
print("DBG0600")
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=qdrant_url,)
vector_store.add_documents(documents=doc)
print("DBG0700")
