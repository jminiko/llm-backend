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

mistral_api_key = os.getenv('MistralAI')
qdrant_uri = os.getenv('Qdrant_URL')
qdrant_api_key = os.getenv('Qdrant_API_KEY')
vectors_size = os.getenv('VectorsSIZE')
collection_name = os.getenv('CollectionIndex')

client = qdrant_client.QdrantClient(
    url=qdrant_uri,
)

if not client.collection_exists(collection_name):
    client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  
    )
    
doc = read_pdf(DIRECTORY_PATH)

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
    url=qdrant_uri,)
vector_store.add_documents(documents=doc)