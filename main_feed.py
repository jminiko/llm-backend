import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
import uuid
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import VectorParams,Distance,PointStruct

load_dotenv()
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name = os.getenv('COLLECTION_NAME')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH"]

openai_embeddings =  MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)

# Defining the model
model = ChatMistralAI(api_key=mistral_api_key,model="mistral-large-latest")

# Defining the qdrant client
qdrant_client = QdrantClient(url=qdrant_url, prefer_grpc=True)

def pdf_path_to_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    langchain_documents = []
    for doc in documents:
        langchain_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "project": "project_1",
                },
            )
        )
    return langchain_documents

def split_pdf(pdf_path, chunk_size=10000, chunk_overlap=0):
    langchain_documents = pdf_path_to_document(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(langchain_documents)
    return texts


def get_or_create_collection():
    if not qdrant_client.collection_exists(collection_name):
        print("create coll")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    
def add_to_qdrant(chunks, embeddings):
    unique_id = str(uuid.uuid4())
    #point = PointStruct(id=unique_id, vector={"default": embeddings}, payload=chunks)
    #info = qdrant_client.upsert(collection_name=collection_name, points=[point])
    qdrant = Qdrant.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        prefer_grpc=True,
    )

def add_unique_id(chunks,file,root):
    for chunk in chunks:
        chunk.metadata = {}
        chunk.metadata["chunk_id"] =str(uuid.uuid4())
        chunk.metadata["file_name"] = file
        chunk.metadata["file_path"] = f"{root}/{file}"
        
    return chunks

def create_chunks():
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if(not root == '.' ):
                print(f"{root}/{file}")
                chunks = split_pdf(f"{root}/{file}")
                add_unique_id(chunks,file,root)
                add_to_qdrant(chunks, openai_embeddings)
                

get_or_create_collection()
chunks = create_chunks()



