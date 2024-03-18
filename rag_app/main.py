from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from embedding_model import Embedder
import chromadb
from fastapi import FastAPI, status, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from rag_base import RetrievalAugmentedGenerator
from retrieval_utils import CrossEncoderReRanker
import uvicorn
from pydantic import BaseModel, Field
from llm_generator import OpenAI_LLMGenerator
import os
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import math
from sentence_transformers import CrossEncoder



_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_PORT=8000
DB_HOST="http://0.0.0.0"
EMBEDDER_NAME="sentence-transformers/all-MiniLM-L12-v2"
TOKENIZER_NAME="sentence-transformers/all-MiniLM-L12-v2"
COLLECTION_NAME="default_collection"

#Promts
SYS_PROMT_GENERATOR="""You are a helpful and knowledgeable advisor 
                        that uses provided information to combine your 
                        knowledge with this info. Be helpful"""

#START UP
db_client = chromadb.HttpClient(host=DB_HOST,
                                port=DB_PORT)
print("Connected to DB...")

embedder = Embedder(model_name=EMBEDDER_NAME,
                    tokenizer_name=TOKENIZER_NAME)
print("Embedding is ready...")




openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_llm_generator = OpenAI_LLMGenerator(openai_client=openai_client,
                                           system_promt=SYS_PROMT_GENERATOR,
                                           max_token=1500,
                                           modelname="gpt-3.5-turbo")
print("LLM generator is ready...")


cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = CrossEncoderReRanker(cross_encoder=cross_encoder)
print("Reranker is ready...")

rag = RetrievalAugmentedGenerator(db_client, embedder,
                                   "default_collection",
                                    reranker=reranker)
print("RAG is started...")

app = FastAPI()
print("Sucessfully started...")

#Data Validation
class RetrieveQuery(BaseModel):
    query: str
    top_k: int

class QA_Query(BaseModel):
    query: str
    top_k: int
    use_query_augmentation : int = Field(default=0)
    use_rerank : int = Field(default=0)



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

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        return {"message": "This endpoint accepts only PDF files."}
    
        # Read file content
    content = file.file.read()  # Directly read without await
        
        # Write file to disk
    with open(file.filename, "wb") as f:
        f.write(content)
        
    print(rag)
        # Assuming rag.upload_pdf_file is synchronous. If it's inherently async, you'd need to adjust its implementation.
    rag.upload_pdf_file(path_file=file.filename, batch_size=5)
        
        # Make sure to close the file and remove it after processing
    file.file.close()
    os.remove(file.filename)

    return {"message": "File added to collection"}


@app.post("/ask")
def generate_answer(query_data : QA_Query):
    query_text = query_data.query
    n_to_retrieve = query_data.top_k



    retrieved_chunks = rag.query_with_text(queries=query_text.split("\n"),
                                            top_k=n_to_retrieve,
                                            use_rerank=query_data.use_rerank)
    
    response = openai_llm_generator.generate_response(query_text=query_text,
                                                      relevant_chunks=retrieved_chunks)
    
    return response

