from datetime import datetime
from zipfile import ZipFile
from dotenv import load_dotenv
import os
import openai
import langchain
import uuid 
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
from streamlit.components.v1 import html
#from htmlTemplates import bot_template, user_template, css
load_dotenv()
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')
collection_name= os.getenv('MUZZO_COLLECTION_NAME')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
vectors_size = os.getenv('VECTOR_SIZE')
hf_token =  os.environ["HF_TOKEN"]
directory_path = os.environ["DIRECTORY_PATH2"]




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


def search_vector_db(query,k):
    doc_search = get_data(query,k)
    return doc_search

def get_answers(query,k):
    doc_search = get_data(query,k)
    
    response = chain.run(input_documents=doc_search, question=query)

    return response


with st.sidebar:
    st.write("abonnez vous pour moins de 30€ / mois et recherchez dans notre base de plus 35 000 CVs")
    html('<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="21talents" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>')
    #add_auth(required=True)
    #after authentication, the email and subscription status is stored in session state
    
    #st.write(f"vous êtes: {st.session_state.email}")
    
#with open(".streamlit/users.txt", "a") as f:
    #current_dateTime = datetime.now()

    #f.write(f"{st.session_state.email} subscribed: {st.session_state.user_subscribed} : {current_dateTime}")
    #f.write("\n")

st.write(unsafe_allow_html=True)

with open(".streamlit/prompt_muzzo.cfg", "r") as f:
    prompt = f.read()
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None



    
st.header("Je suis ton assistant IA pour t'aider à trouver des CV sur muzzo")

# show user input
with st.form("my_form"):
    option = st.selectbox(
        "Nombre de CV en retour",
        ("", "10", "20","50","100"),
    )


    zips = []
    user_question = st.text_area("Pose ta question:",value=prompt,height=300)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Aller...")
    if submitted:
        
        

        zips = []
        if user_question:
            st.write(f"Question: {user_question}")
            if(option==""):
                option = 5

        answer = get_answers(user_question,option)
        st.write(f"Answer: {answer}")
        docs = embeddings.embed_query(user_question)
    
        answer =  search_vector_db(user_question,option)

        for doc in answer:
            path = doc.dict()['metadata']['file_name']
            path = "/root/SamIA/hw/muzzo/processed/"+path #path.replace('resume_main','main')
            zips.append(path)
#                pdf_viewer(input=path,key=doc.dict()['metadata']['chunk_id'],width=700)
if(len(zips)!=0):
    unique_id = str(uuid.uuid4())
    
    with ZipFile(f"resultat-{unique_id}.zip",'w') as zip:
        # writing each file one by one 
        for file in zips:
            abs_src = os.path.abspath(file)
            arcname = os.path.basename(abs_src) 
            zip.write(file,arcname)
    with open(f"resultat-{unique_id}.zip", "rb") as fp:
        btn = st.download_button(
            label="Download ZIP",
            data=fp,
            file_name=f"myfile-{unique_id}.zip",
            mime="application/zip"
        )
        st.write(btn)

