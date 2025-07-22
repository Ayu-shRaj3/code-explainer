import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# --- SETUP ---
st.set_page_config(page_title="AI Code Explainer", layout="centered")

# Title
st.title("🧠 AI Code Explainer (Local)")

# Sidebar
st.sidebar.header("📁 Project Info")
st.sidebar.markdown("**LLM:** phi (via Ollama)")
st.sidebar.markdown("**Embeddings:** MiniLM (HuggingFace)")
st.sidebar.markdown("**Storage:** Local (ChromaDB)")

# --- Load Embeddings & Vector Store ---
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory="data/embeddings",
        embedding_function=embedding_model,
        collection_name="code_docs"
    )

    llm = ChatOllama(model="phi")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

qa_chain = load_chain()

# --- Main App ---
user_question = st.text_input("💬 Ask something about your code/doc:", placeholder="e.g. What does greet() do?")

if user_question:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": user_question})
        st.markdown("### 📘 Answer:")
        st.success(result["result"])

        # Optional: show sources
        with st.expander("📄 Sources used"):
            for doc in result["source_documents"]:
                st.text(doc.metadata.get("source", "from memory"))
                st.write(doc.page_content[:500] + "...")
