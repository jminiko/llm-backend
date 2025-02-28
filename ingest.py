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
load_dotenv()

qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name = os.getenv('COLLECTION_NAME')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH"]

client = qdrant_client.QdrantClient(
    url=qdrant_url,
)

if not client.collection_exists(collection_name):
    client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  
    )
def read_pdf(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
    
doc = read_pdf(directory_path)

len(doc)

def chunk_data(docs, chunk_size = 1500, chunk_overlap = 10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap= chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc
documents = chunk_data(docs = doc)


embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=qdrant_url,)
vector_store.add_documents(documents=doc)