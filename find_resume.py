from dotenv import load_dotenv
import os
import openai
import langchain
from uuid import uuid4
import streamlit as st
from st_paywall import add_auth
import qdrant_client
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams,Distance
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai.chat_models import ChatMistralAI
from streamlit_pdf_viewer import pdf_viewer
#from htmlTemplates import bot_template, user_template, css
load_dotenv()
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name = os.getenv('COLLECTION_NAME')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH"]

def read_pdf(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=qdrant_url,)

retriever = vector_store.as_retriever()


def get_data(query, k=2):
    matching_results = vector_store.similarity_search(query, k=k)
    return matching_results

llm= ChatMistralAI(api_key=mistral_api_key,model="mistral-large-latest")
chain = load_qa_chain(llm, chain_type="stuff")


def search_vdb(query):
    doc_search = get_data(query)
    return doc_search

def get_answers(query):
    doc_search = get_data(query)
    
    response = chain.run(input_documents=doc_search, question=query)

    return response,doc_search

def main():
    load_dotenv()
    add_auth(required=True)

    #after authentication, the email and subscription status is stored in session state
    st.write(st.session_state.email)
    st.write(st.session_state.user_subscribed)
    

    st.write(unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    

    st.set_page_config(page_title="Samia CV")
    st.header("Je suis ton assistant IA pour t'aider à trouver des CV.💭")
    
    # show user input
    user_question = st.text_input("Pose ta question:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = get_answers(user_question)
        
        st.write(f"Answer: {answer}")
        docs = embeddings.embed_query(user_question)
        
        answer =  search_vdb(user_question)
        for doc in answer:
            pdf_viewer(doc.dict()['metadata']['source'],
                   width=700)

if __name__ == '__main__':
    main()
