from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# Load local embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store from disk
db = Chroma(
    persist_directory="data/embeddings",
    embedding_function=embedding_model,
    collection_name="code_docs"
)

# Use local LLM via Ollama
llm = ChatOllama(model="phi")  # You can also use "tinyllama", "gemma:2b", etc.

# Set up retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Ask a question
query = input("🧠 Ask a question about your code/doc: ")
result = qa_chain.invoke({"query": query})


# Show the answer
print("\n📘 Answer:")
print(result["result"])
