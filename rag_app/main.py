
import os
from embedding_model import Embedder
import chromadb
from fastapi import FastAPI, status, File, UploadFile, HTTPException
from rag_base import RetrievalAPI
from pydantic import BaseModel, Field
from llm_generator import OpenAI_LLMGenerator
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


#CONSTANTS 
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_PORT=8000
DB_HOST="chroma"
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

rag = RetrievalAPI(db_client, embedder, "default_collection",)
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
    
    content = file.file.read()
    with open(file.filename, "wb") as f:
        f.write(content)
        
    print(rag)
    rag.upload_pdf_file(path_file=file.filename, batch_size=5)
        
    file.file.close()
    os.remove(file.filename)

    return {"message": "File added to collection"}


# @app.post("/ask")
# def generate_answer(query_data : QA_Query):
#     query_text = query_data.query
#     n_to_retrieve = query_data.top_k

#     retrieved_chunks = rag.query_with_text(queries=query_text.split("\n"),
#                                             top_k=n_to_retrieve,
#                                             use_rerank=query_data.use_rerank)
    
#     response = openai_llm_generator.generate_response(query_text=query_text,
#                                                       relevant_chunks=retrieved_chunks)
    
#     return response

