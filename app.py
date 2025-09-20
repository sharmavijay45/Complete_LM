# # app.py
# import os
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from typing import List

# # Optional Groq integration (only used if GROQ_API_KEY is provided)
# try:
#     from groq import Groq
#     _has_groq = True
# except Exception:
#     _has_groq = False

# # Config
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# # UI
# st.set_page_config(page_title="RAG Demo (Qdrant + Groq)", layout="wide")
# st.title("RAG Demo — Qdrant + Groq (optional)")

# st.sidebar.markdown("### Settings")
# qdrant_url = st.sidebar.text_input("Qdrant URL", QDRANT_URL)
# qdrant_api_key = st.sidebar.text_input("Qdrant API Key (optional)", QDRANT_API_KEY)
# collection = st.sidebar.text_input("Collection name", COLLECTION_NAME)
# model_name = st.sidebar.text_input("Embed model", EMBED_MODEL)
# top_k = st.sidebar.slider("Top k results", 1, 20, 5)

# # load resources with caching
# @st.cache_resource
# def load_model(name: str):
#     return SentenceTransformer(name)

# @st.cache_resource
# def load_qdrant(url: str, api_key: str):
#     if api_key:
#         return QdrantClient(url=url, api_key=api_key)
#     return QdrantClient(url=url)

# model = load_model(model_name)
# client = load_qdrant(qdrant_url, qdrant_api_key)

# groq_client = None
# if GROQ_API_KEY and _has_groq:
#     groq_client = Groq(api_key=GROQ_API_KEY)

# def retrieve_docs(query: str, k: int = 5) -> List[dict]:
#     vec = model.encode(query).tolist()
#     results = client.search(collection_name=collection, query_vector=vec, limit=k)
#     # Each result has .payload and .score
#     docs = []
#     for r in results:
#         docs.append({"content": r.payload.get("content"), "file": r.payload.get("file"), "score": getattr(r, "score", None)})
#     return docs

# def ask_groq(query: str, context: str):
#     if not groq_client:
#         return "Groq client not configured or 'groq' package not installed."
#     prompt = f"""You are a helpful assistant. Use the following context to answer the question.

# Context:
# {context}

# Question:
# {query}

# Answer concisely."""
#     resp = groq_client.chat.completions.create(
#         model="llama-3.1-70b-versatile",
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return resp.choices[0].message["content"]

# # Main UI
# query = st.text_input("Enter your question here:")

# if st.button("Search / Ask") or (query and st.session_state.get("auto_run", False)):
#     if not query.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Retrieving relevant chunks..."):
#             docs = retrieve_docs(query, k=top_k)
#         st.subheader("Retrieved chunks")
#         for i, d in enumerate(docs, 1):
#             st.markdown(f"**{i}. file:** `{d['file']}`  \n**score:** {d['score']}\n\n{d['content'][:1000]}")  # show truncated

#         # Optionally call Groq (if configured)
#         if GROQ_API_KEY and _has_groq:
#             with st.spinner("Generating answer from Groq LLM..."):
#                 context = "\n\n".join([d["content"] for d in docs])
#                 answer = ask_groq(query, context)
#             st.subheader("Groq Answer")
#             st.write(answer)
#         else:
#             st.info("Groq not configured or not installed. Only showing retrieved context.")





# import os
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from typing import List
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # Config
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# # UI
# st.set_page_config(page_title="RAG Demo (Qdrant + Groq)", layout="wide")
# st.title("RAG Demo — Qdrant + Groq (optional)")

# st.sidebar.markdown("### Settings")
# qdrant_url = st.sidebar.text_input("Qdrant URL", QDRANT_URL)
# qdrant_api_key = st.sidebar.text_input("Qdrant API Key (optional)", QDRANT_API_KEY)
# collection = st.sidebar.text_input("Collection name", COLLECTION_NAME)
# model_name = st.sidebar.text_input("Embed model", EMBED_MODEL)
# top_k = st.sidebar.slider("Top k results", 1, 20, 5)

# # Temporary debug: Check if key loaded (remove after testing)
# st.sidebar.write(f"Debug: GROQ_API_KEY loaded? {'Yes' if GROQ_API_KEY else 'No'}")

# # load resources with caching
# @st.cache_resource
# def load_model(name: str):
#     return SentenceTransformer(name)

# @st.cache_resource
# def load_qdrant(url: str, api_key: str):
#     if api_key:
#         return QdrantClient(url=url, api_key=api_key)
#     return QdrantClient(url=url)

# model = load_model(model_name)
# client = load_qdrant(qdrant_url, qdrant_api_key)

# groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY else None

# def retrieve_docs(query: str, k: int = 5) -> List[dict]:
#     vec = model.encode(query).tolist()
#     results = client.search(collection_name=collection, query_vector=vec, limit=k)
#     # Each result has .payload and .score
#     docs = []
#     for r in results:
#         docs.append({"content": r.payload.get("content"), "file": r.payload.get("file"), "score": getattr(r, "score", None)})
#     return docs

# def ask_groq(query: str, context: str):
#     if not groq_client:
#         return "Groq client not configured."
#     from langchain_core.messages import HumanMessage, SystemMessage
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Answer the question ONLY using information from the provided context below. Do not add any external knowledge or assumptions. Include all relevant details from the context comprehensively."),
#         SystemMessage(content=f"Context:\n{context}"),
#         HumanMessage(content=f"Question:\n{query}")
#     ]
#     response = groq_client.invoke(messages)
#     return response.content

# # Main UI
# query = st.text_input("Enter your question here:")

# if st.button("Search / Ask") or (query and st.session_state.get("auto_run", False)):
#     if not query.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Retrieving relevant chunks..."):
#             docs = retrieve_docs(query, k=top_k)
#         st.subheader("Retrieved chunks")
#         for i, d in enumerate(docs, 1):
#             st.markdown(f"**{i}. file:** `{d['file']}`  \n**score:** {d['score']}\n\n{d['content'][:1000]}")  # show truncated

#         # Optionally call Groq (if configured)
#         if GROQ_API_KEY:
#             with st.spinner("Generating answer from Groq LLM..."):
#                 context = "\n\n".join([d["content"] for d in docs])
#                 answer = ask_groq(query, context)
#             st.subheader("Groq Answer")
#             st.write(answer)
#         else:
#             st.info("Groq not configured. Only showing retrieved context.")




# import os
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from typing import List
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # Config
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# # UI
# st.set_page_config(page_title="RAG Demo (Qdrant + Groq)", layout="wide")
# st.title("RAG Demo — Qdrant + Groq (optional)")

# st.sidebar.markdown("### Settings")
# qdrant_url = st.sidebar.text_input("Qdrant URL", QDRANT_URL)
# qdrant_api_key = st.sidebar.text_input("Qdrant API Key (optional)", QDRANT_API_KEY)
# collection = st.sidebar.text_input("Collection name", COLLECTION_NAME)
# model_name = st.sidebar.text_input("Embed model", EMBED_MODEL)
# top_k = st.sidebar.slider("Top k results", 1, 20, 5)

# # Temporary debug: Check if key loaded (remove after testing)
# st.sidebar.write(f"Debug: GROQ_API_KEY loaded? {'Yes' if GROQ_API_KEY else 'No'}")

# # load resources with caching
# @st.cache_resource
# def load_model(name: str):
#     return SentenceTransformer(name)

# @st.cache_resource
# def load_qdrant(url: str, api_key: str):
#     if api_key:
#         return QdrantClient(url=url, api_key=api_key)
#     return QdrantClient(url=url)

# model = load_model(model_name)
# client = load_qdrant(qdrant_url, qdrant_api_key)

# groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY else None

# def retrieve_docs(query: str, k: int = 5) -> List[dict]:
#     vec = model.encode(query).tolist()
#     results = client.query_points(collection_name=collection, query=vec, limit=k).points
#     # Each result has .payload and .score
#     docs = []
#     for r in results:
#         docs.append({"content": r.payload.get("content"), "file": r.payload.get("file"), "score": getattr(r, "score", None)})
#     return docs

# def ask_groq(query: str, context: str):
#     if not groq_client:
#         return "Groq client not configured."
#     from langchain_core.messages import HumanMessage, SystemMessage
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Answer the question ONLY using information from the provided context below. Do not add any external knowledge or assumptions. Include all relevant details from the context comprehensively."),
#         SystemMessage(content=f"Context:\n{context}"),
#         HumanMessage(content=f"Question:\n{query}")
#     ]
#     response = groq_client.invoke(messages)
#     return response.content

# # Main UI
# query = st.text_input("Enter your question here:")

# if st.button("Search / Ask") or (query and st.session_state.get("auto_run", False)):
#     if not query.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Retrieving relevant chunks..."):
#             docs = retrieve_docs(query, k=top_k)
#         st.subheader("Retrieved chunks")
#         for i, d in enumerate(docs, 1):
#             st.markdown(f"**{i}. file:** `{d['file']}`  \n**score:** {d['score']}\n\n{d['content'][:1000]}")  # show truncated

#         # Optionally call Groq (if configured)
#         if GROQ_API_KEY:
#             with st.spinner("Generating answer from Groq LLM..."):
#                 context = "\n\n".join([d["content"] for d in docs])
#                 answer = ask_groq(query, context)
#             st.subheader("Groq Answer")
#             st.write(answer)
#         else:
#             st.info("Groq not configured. Only showing retrieved context.")










# import os
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from typing import List
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from fastapi import FastAPI, Body
# from pydantic import BaseModel

# # Load environment variables from .env
# load_dotenv()

# # Config
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# # Load resources
# model = SentenceTransformer(EMBED_MODEL)
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)

# groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY else None

# def retrieve_docs(query: str, k: int = 5) -> List[dict]:
#     vec = model.encode(query).tolist()
#     results = client.query_points(collection_name=COLLECTION_NAME, query=vec, limit=k).points
#     # Each result has .payload and .score
#     docs = []
#     for r in results:
#         docs.append({"content": r.payload.get("content"), "file": r.payload.get("file"), "score": getattr(r, "score", None)})
#     return docs

# def ask_groq(query: str, context: str):
#     if not groq_client:
#         return "Groq client not configured."
#     from langchain_core.messages import HumanMessage, SystemMessage
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Answer the question ONLY using information from the provided context below. Do not add any external knowledge or assumptions. Include all relevant details from the context comprehensively."),
#         SystemMessage(content=f"Context:\n{context}"),
#         HumanMessage(content=f"Question:\n{query}")
#     ]
#     response = groq_client.invoke(messages)
#     return response.content

# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 5

# @app.post("/rag")
# def rag_query(request: QueryRequest):
#     if not request.query.strip():
#         return {"error": "Please enter a question."}
    
#     docs = retrieve_docs(request.query, request.top_k)
    
#     retrieved_chunks = []
#     for i, d in enumerate(docs, 1):
#         retrieved_chunks.append({
#             "index": i,
#             "file": d['file'],
#             "score": d['score'],
#             "content": d['content'][:1000]  # truncated as in original
#         })
    
#     answer = ""
#     if GROQ_API_KEY:
#         context = "\n\n".join([d["content"] for d in docs])
#         answer = ask_groq(request.query, context)
#     else:
#         answer = "Groq not configured. Only showing retrieved context."
    
#     return {
#         "retrieved_chunks": retrieved_chunks,
#         "groq_answer": answer
#     }



# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from typing import List
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Config
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# # Load resources
# model = SentenceTransformer(EMBED_MODEL)
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
# groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY else None

# # FastAPI app
# app = FastAPI()

# # ✅ Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or restrict to ["http://127.0.0.1:5500"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 5

# def retrieve_docs(query: str, k: int = 5) -> List[dict]:
#     vec = model.encode(query).tolist()
#     results = client.query_points(collection_name=COLLECTION_NAME, query=vec, limit=k).points
#     docs = []
#     for r in results:
#         docs.append({
#             "content": r.payload.get("content"),
#             "file": r.payload.get("file"),
#             "score": getattr(r, "score", None)
#         })
#     return docs

# def ask_groq(query: str, context: str):
#     if not groq_client:
#         return "Groq client not configured."
#     from langchain_core.messages import HumanMessage, SystemMessage
#     messages = [
#         SystemMessage(content="You are a helpful assistant. Answer ONLY using context."),
#         SystemMessage(content=f"Context:\n{context}"),
#         HumanMessage(content=f"Question:\n{query}")
#     ]
#     response = groq_client.invoke(messages)
#     return response.content

# @app.post("/rag")
# def rag_query(request: QueryRequest):
#     if not request.query.strip():
#         return {"error": "Please enter a question."}
    
#     docs = retrieve_docs(request.query, request.top_k)
#     retrieved_chunks = []
#     for i, d in enumerate(docs, 1):
#         retrieved_chunks.append({
#             "index": i,
#             "file": d['file'],
#             "score": d['score'],
#             "content": d['content'][:1000]
#         })
    
#     answer = ""
#     if GROQ_API_KEY:
#         context = "\n\n".join([d["content"] for d in docs if d["content"]])
#         answer = ask_groq(request.query, context)
#     else:
#         answer = "Groq not configured. Only showing retrieved context."
    
#     return {
#         "retrieved_chunks": retrieved_chunks,
#         "groq_answer": answer
#     }






from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
MONGO_URI = os.getenv("MONGODB_URI")

# Load resources
model = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)
groq_client = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY else None

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["LM"]  # explicit database name
history_collection = db["chat_history"]

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class HistoryItem(BaseModel):
    query: str
    groq_answer: str = ""
    retrieved_chunks: List[dict] = []

# RAG functions
def retrieve_docs(query: str, k: int = 5) -> List[dict]:
    vec = model.encode(query).tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=vec, limit=k).points
    docs = []
    for r in results:
        docs.append({
            "content": r.payload.get("content"),
            "file": r.payload.get("file"),
            "score": getattr(r, "score", None)
        })
    return docs

def ask_groq(query: str, context: str):
    if not groq_client:
        return "Groq client not configured."
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer ONLY using context."),
        SystemMessage(content=f"Context:\n{context}"),
        HumanMessage(content=f"Question:\n{query}")
    ]
    response = groq_client.invoke(messages)
    return response.content

# Endpoints
@app.post("/rag")
def rag_query(request: QueryRequest):
    if not request.query.strip():
        return {"error": "Please enter a question."}
    
    docs = retrieve_docs(request.query, request.top_k)
    retrieved_chunks = []
    for i, d in enumerate(docs, 1):
        retrieved_chunks.append({
            "index": i,
            "file": d['file'],
            "score": d['score'],
            "content": d['content'][:1000]
        })
    
    answer = ""
    if GROQ_API_KEY:
        context = "\n\n".join([d["content"] for d in docs if d["content"]])
        answer = ask_groq(request.query, context)
    else:
        answer = "Groq not configured. Only showing retrieved context."

    # Save to MongoDB history
    history_collection.insert_one({
    "query": request.query,
    "groq_answer": answer,
    "retrieved_chunks": retrieved_chunks,
    "timestamp": datetime.datetime.utcnow()
    })
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "groq_answer": answer
    }

@app.get("/history")
def get_history():
    chats = list(history_collection.find({}, {"_id":0}).sort("timestamp", 1))  # chronological
    return chats
