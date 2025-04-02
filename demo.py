import os
import time
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

os.environ["GOOGLE_API_KEY"] = "AIzaSyCgDPB0WnLxg2fGb13M8dVq833lThkdyXM"


gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini()

Settings.llm = llm
Settings.embed_model = gemini_embedding_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 2080
Settings.context_window = 3900

persist_dir = "./storage"
doc_folder = "doc"

def get_latest_modification_time(directory):
    """Get the latest modification time of files in the directory."""
    return max(os.path.getmtime(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith(".pdf"))

# Check if index exists and if files are updated
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("No existing index found. Creating a new one...")
    index = None
    last_modified = None
else:
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    last_modified = get_latest_modification_time(doc_folder)

while True:
    current_mod_time = get_latest_modification_time(doc_folder)
    
    if last_modified is None or current_mod_time > last_modified:
        print("New or modified documents detected. Rebuilding index...")
        documents = SimpleDirectoryReader(doc_folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist()
        last_modified = current_mod_time
    else:
        print("Using existing index.")

    # Create Query Engine
    query_engine = index.as_query_engine()

    # Interactive Question-Answering Loop
    user_input = input("Ask a question (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = query_engine.query(user_input)
    print("\nAnswer:", response.response, "\n")
