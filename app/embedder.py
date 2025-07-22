from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

def embed_docs():
    loader = TextLoader("data/doc.txt", encoding="utf-8")
  # Ensure this file exists
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory="data/embeddings",
        embedding_function=embedding_model,
        collection_name="code_docs"
    )

    db.add_documents(split_docs)
    db.persist()

embed_docs()

