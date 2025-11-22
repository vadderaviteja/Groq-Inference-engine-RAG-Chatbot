
# Groq LLM RAG Chatbot using Streamlit, FAISS & LangChain
## 🚀 Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built using:

Groq Llama-3.3-70B Versatile model (via ChatGroq)

FAISS vector database

Ollama embeddings

LangChain Runnables pipeline

Streamlit UI

The chatbot fetches real knowledge from websites, converts it into embeddings, stores them in FAISS, and then uses Groq's ultra-fast inference to answer queries accurately.

# How the System Works (Mechanism)
## 1️⃣ Document Loading

The project loads data from a URL:

WebBaseLoader("https://www.geeksforgeeks.org/machine-learning/machine-learning/")


The page content is extracted and prepared for indexing.

## 2️⃣ Text Splitting

Large text is split into smaller chunks using:

RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


This ensures embeddings are meaningful.

## 3️⃣ Embedding Generation

Ollama Embeddings create numerical representations of each text chunk:

OllamaEmbeddings(model="llama3")


These embeddings help the system find relevant information from the user's query.

## 4️⃣ Vector Storage (FAISS)

The embeddings are stored in a FAISS vector database:

FAISS.from_documents(final_documents, embeddings)


FAISS performs fast similarity search.

## 5️⃣ Retrieval

When a user asks a question, FAISS retrieves the most relevant chunks:

retriever = vectors.as_retriever()

## 6️⃣ RAG Pipeline (LangChain Runnable)

This is the heart of your project:

document_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# This pipeline performs:

## Step	Action
1	Retrieve relevant text

2	Combine question + context

3	Send to Groq Llama model

4	Parse answer

# 7️⃣ LLM Response

The user’s question + retrieved knowledge is passed to:

llm = ChatGroq(model_name="llama-3.3-70b-versatile")


Groq returns an accurate answer in milliseconds.

# 8️⃣ Streamlit UI

Streamlit provides a clean interface for interacting with the chatbot:

Input prompt

Response display

Document similarity expander

# Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Create virtual environment

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

4. Install dependencies

streamlit

langchain

langchain-core

langchain-community

langchain-groq

langchain-ollama

faiss-cpu

python-dotenv

requests


pip install -r requirements.txt

6. Run the app
streamlit run main.py

# 🔐 API Key Setup

Inside your code:

groq_api_key = "your-api-key"


Or use a .env file:

GROQ_API_KEY=your-api-key

# Features

✅ Retrieval-Augmented Generation

✅ Fast Llama-3.3-70B inference via Groq

✅ FAISS-based vector search

✅ Ollama embeddings

✅ Fully working Streamlit UI

✅ Real-time document similarity viewer
