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


def get_data(query, k=5):
    matching_results = vector_store.similarity_search(query, k=k)
    return matching_results

llm= ChatMistralAI(api_key=mistral_api_key,model="mistral-large-latest")
chain = load_qa_chain(llm, chain_type="stuff")


def search_vector_db(query):
    doc_search = get_data(query)
    return doc_search

def get_answers(query):
    doc_search = get_data(query)
    
    response = chain.run(input_documents=doc_search, question=query)

    return response

def main():
    load_dotenv()

    
    add_auth(required=True)    
    #after authentication, the email and subscription status is stored in session state
    st.write(f"vous êtes: {st.session_state.email}")
    st.write("La recherche est limitée à 5 CV maximum")
    #st.write(st.session_state.user_subscribed)
    with open(".streamlit/users.txt", "w") as f:
        f.write(f"{st.session_state.email} subscribed: {st.session_state.user_subscribed}")
    
    st.write(unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    


    st.header("Je suis ton assistant IA pour t'aider à trouver des CV.💭")
    
    # show user input
    with st.form("my_form"):

        user_question = st.text_area("Pose ta question:",value="Vous êtes Alex Recruteur, un expert en recrutement avec 20 ans d'expérience.\n Votre spécialité est d'analyser les fiches de poste et les CV pour déterminer les correspondances entre les compétences des candidats et les exigences des postes.\n Je recherche des profils de : ",height=300)
        # Every form must have a submit button.
        submitted = st.form_submit_button("Aller...")
        if submitted:
            if user_question:
                st.write(f"Question: {user_question}")
                answer = get_answers(user_question)
        
                st.write(f"Answer: {answer}")
                docs = embeddings.embed_query(user_question)
        
                answer =  search_vector_db(user_question)
                for doc in answer:
                    path = doc.dict()['metadata']['file_path']
                    path = path.replace('resume_main','main')
                    print(path)
                    pdf_viewer(input=path,key=doc.dict()['metadata']['chunk_id'],width=700)

if __name__ == '__main__':
    main()
