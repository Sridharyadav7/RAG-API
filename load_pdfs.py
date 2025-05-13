# load_pdfs.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def load_and_split_pdfs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".pdf", ".txt")):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            pages = loader.load()
            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(all_docs)

def index_documents(folder_path):
    documents = load_and_split_pdfs(folder_path)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    PineconeClient(api_key=PINECONE_API_KEY)

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        index_name=PINECONE_INDEX_NAME,
    )
    #print("✅ Documents indexed successfully!")

if __name__ == "__main__":
    index_documents("./pdfs")
