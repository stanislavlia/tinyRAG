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

import math


DB_PORT=8000
DB_HOST="localhost"
EMBEDDER_NAME="sentence-transformers/all-MiniLM-L12-v2"
TOKENIZER_NAME="sentence-transformers/all-MiniLM-L12-v2"

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
def retrieve(query_data):

    print(query_data)
    query_dict = query_data.dict()

    query_text = query_dict["query"]
    n_to_retrive = int(query_dict["top_k"])

    response = rag.query_with_text(queries=query_text.split("\n"),
                                   top_k=n_to_retrive)
    
    return response
