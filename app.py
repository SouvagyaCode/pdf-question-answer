import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ["GOOGLE_API_KEY"] = API_KEY


DOC_FOLDER = "doc"

if not os.path.exists(DOC_FOLDER):
    os.makedirs(DOC_FOLDER)

# Initialize models
gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini()

# Configure settings
Settings.llm = llm
Settings.embed_model = gemini_embedding_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 2080
Settings.context_window = 3900

def save_uploadedfile(uploadedfile):

    if os.path.exists(DOC_FOLDER):
        shutil.rmtree(DOC_FOLDER) 
    os.makedirs(DOC_FOLDER)

    file_path = os.path.join(DOC_FOLDER, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def get_latest_modification_time(directory):

    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    return max(os.path.getmtime(os.path.join(directory, f)) for f in pdf_files)

st.title("Gemini PDF Query App")
st.write("Upload a PDF and ask questions about its content.")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    file_path = save_uploadedfile(uploaded_pdf)
    st.success(f"File '{uploaded_pdf.name}' uploaded successfully!")


    documents = SimpleDirectoryReader(DOC_FOLDER).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    query_engine = index.as_query_engine()

    # User input for queries
    user_query = st.text_input("Enter your question:")

    if user_query:
        response = query_engine.query(user_query)
        st.markdown(f"**Answer:** {response.response}")

