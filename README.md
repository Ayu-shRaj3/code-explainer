# 🧠 AI-Powered Code Explainer

This project is an **AI-powered code explanation tool** that uses **local LLMs** (like `phi` via Ollama) and **retrieval-augmented generation (RAG)** to help users understand code snippets and built-in Python functions.

Built with:
- ✅ Local LLMs (via [Ollama](https://ollama.com))
- ✅ LangChain RAG pipeline
- ✅ Local vector search using ChromaDB
- ✅ Frontend using Streamlit
- ✅ Embeddings from Sentence-Transformers

---

## 🚀 Features

- 🔍 Ask questions like _"What does `range()` do?"_ or _"Explain this Python snippet"_
- 📁 Embeds Python documentation and retrieves relevant context before answering
- ⚙️ Runs fully offline using local LLMs like `phi`, `tinyllama`, or `mistral` (via Ollama)
- 🧠 Uses `sentence-transformers` for semantic similarity
- 🖥️ Easy-to-use Streamlit interface

---

## 🛠️ Tech Stack

| Layer         | Technology                  |
|---------------|-----------------------------|
| LLM           | Ollama (models like `phi`, `tinyllama`) |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB     | ChromaDB                    |
| Backend       | LangChain                   |
| UI            | Streamlit                   |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ayu-shRaj3/code-explainer.git
cd code-explainer
