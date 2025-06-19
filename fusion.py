import chunk
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
from dotenv import load_dotenv
import os
from langchain_mistralai import MistralAIEmbeddings
from datetime import datetime
from qdrant_client.models import VectorParams,Distance
from qdrant_client import AsyncQdrantClient
import asyncio
# Configuration
QDRANT_HOST = "localhost"  # ou l'URL de votre serveur
QDRANT_PORT = 6333
COLLECTION_MAIN = "target-final"
COLLECTION_FINAL = "index-resume-final"
COLLECTION_SOURCE = "index-resume-muzzo"
BATCH_SIZE = 40
CHUNK_SIZE = 100
load_dotenv()
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name = os.getenv('COLLECTION_NAME_FINAL')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH"]

openai_embeddings =  MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
# Connexion au client Qdrant
client = AsyncQdrantClient(url="http://localhost:6333")

collection_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
current_dateTime = datetime.now()

async def fetch_all_points():
    """Récupère tous les points d'une collection Qdrant."""
    scroll_offset = None
    all_points = []
    index = 0
    while True:
        
        response = await client.scroll(
            collection_name=COLLECTION_FINAL,
            offset=scroll_offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )
        
        if not response[0]:
            break
        #all_points.extend(response[0])
        scroll_offset = response[1]
        unique_points = []
        chunked_points = response[0]
        for point in chunked_points:
            payload = point.payload
            payload.get("metadata")["date"] = current_dateTime
            id = point.id
            vector = point.vector
            unique_points.append(PointStruct(id=id, vector=vector, payload=payload))
        await client.upsert(
                collection_name=COLLECTION_MAIN,
                points=unique_points
            )
    print("all points : ", len(all_points))     
    return all_points

async def fetch_all_points_muzzo():
    """Récupère tous les points d'une collection Qdrant."""
    scroll_offset = None
    all_points = []
    index = 0
    while True:
        
        response = await client.scroll(
            collection_name=COLLECTION_SOURCE,
            offset=scroll_offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )
        
        if not response[0]:
            break
        #all_points.extend(response[0])
        scroll_offset = response[1]
        unique_points = []
        chunked_points = response[0]
        for point in chunked_points:
            payload = point.payload
            id = point.id
            vector = point.vector
            unique_points.append(PointStruct(id=id, vector=vector, payload=payload))
        await client.upsert(
                collection_name=COLLECTION_MAIN,
                points=unique_points
            )
    print("all points : ", len(all_points))     
    return all_points

def create_collection():
    if not collection_client.collection_exists(COLLECTION_MAIN):
        collection_client.create_collection(
            collection_name=COLLECTION_MAIN,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

if __name__ == "__main__":
    create_collection()
    #asyncio.run(fetch_all_points())
    asyncio.run(fetch_all_points_muzzo())
    print("Fusion terminée.")

