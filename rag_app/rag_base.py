from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from embedding_model import Embedder
import chromadb
import torch
import hashlib
from retrieval_utils import PdfChunksLoader_ChromaDB

import math



class RetrievalAPI():
    def __init__(self, db_client, embedder, collection_name):
        
        self.db_client = db_client
        self.embedder = embedder
        
        self.collection_name = collection_name
        self.collection = self.db_client.get_or_create_collection(name=self.collection_name)
        
        self.chunk_loader = PdfChunksLoader_ChromaDB(self.collection,
                                                     embedder)
        

    def upload_pdf_file(self, path_file, batch_size=5):
        
        docs = self.chunk_loader._extract_pdf_chunks(path_file)
        
        for i in tqdm(range(math.ceil(len(docs) / batch_size)), desc=f"[{path_file}] loading batches:"):
            self.chunk_loader.populate(docs[i * batch_size : (i + 1) * batch_size])
        
        
        print(f"[{path_file}]: All batches loaded successfully...")
    
    def query_with_embeddings(self, embeddings, top_k):
        
        return self.collection.query(query_embeddings=embeddings,
                                      n_results=top_k)
    
    def query_with_text(self, queries, top_k, use_rerank=False):
                
        embeddings_tensor = self.embedder.compute_embeddings(queries)
        embeddings_list = embeddings_tensor.tolist()
        retrived_docs = self.query_with_embeddings(embeddings_list, top_k)

        return retrived_docs
    
    def get(self, ids, where, limit):
        pass
    
    def reset_collection(self):
        pass
    
    
    
    