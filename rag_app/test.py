from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from embedding_model import Embedder
import chromadb
from fastapi import FastAPI, status, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from rag_base import RetrievalAPI
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
DB_HOST="localhost" #for test 
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

rag = RetrievalAPI(db_client, embedder,
                                   "default_collection",
                                    )
print("RAG is started...")

app = FastAPI()
print("Sucessfully started...")

#Data Validation
class RetrieveQuery(BaseModel):
    query: str
    top_k: int
    use_rerank : int = Field(default=0)


class QA_Query(BaseModel):
    query: str
    top_k: int
    use_query_augmentation : int = Field(default=0)
    use_rerank : int = Field(default=0)

print(rag.query_with_text("Who is Stanislav Liashkov?", use_rerank=False, top_k=3))


