from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from embedding_model import Embedder
import chromadb
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from rag_base import RetrievalAugmentedGenerator
import uvicorn
from pydantic import BaseModel

import math


DB_PORT=8000
DB_HOST="localhost"
EMBEDDER_NAME="sentence-transformers/all-MiniLM-L12-v2"
TOKENIZER_NAME="sentence-transformers/all-MiniLM-L12-v2"
COLLECTION_NAME="default_collection"

#Start up
db_client = chromadb.HttpClient(host=DB_HOST,
                                port=DB_PORT)
print("Connected to DB...")

embedder = Embedder(model_name=EMBEDDER_NAME,
                    tokenizer_name=TOKENIZER_NAME)
print("Embedding is ready...")

rag = RetrievalAugmentedGenerator(db_client, embedder, "default_collection")
print("RAG is started...")

app = FastAPI()

print("Sucessfully started...")

class RetrieveQuery(BaseModel):
    query: str
    top_k: int



@app.get("/")
def home():
    return {
        "Message": "tinyRAG API is running",
        "Health Check ": "OK",
        "Version": "0.0.1",
        "tokenizer" : TOKENIZER_NAME,
        "embedding_model" : EMBEDDER_NAME,

    }

@app.post("/retrieve")
def retrieve(query_data: RetrieveQuery):
    query_text = query_data.query
    n_to_retrieve = query_data.top_k

    response = rag.query_with_text(queries=query_text.split("\n"), top_k=n_to_retrieve)
    
    return response