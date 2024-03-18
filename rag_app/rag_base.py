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



class RetrievalAugmentedGenerator():
    def __init__(self, db_client, embedder, collection_name, reranker=None):
        
        self.db_client = db_client
        self.embedder = embedder
        
        self.collection_name = collection_name
        self.collection = self.db_client.get_or_create_collection(name=self.collection_name)
        
        self.chunk_loader = PdfChunksLoader_ChromaDB(self.collection,
                                                     embedder)
        
        self.reranker = reranker

    def upload_pdf_file(self, path_file, batch_size=5):
        ##Load chunks by batches
        
        docs = self.chunk_loader._extract_pdf_chunks(path_file)
        
        for i in tqdm(range(math.ceil(len(docs) / batch_size)), desc=f"[{path_file}] loading batches:"):
            self.chunk_loader.populate(docs[i * batch_size : (i + 1) * batch_size])
        
        
        print(f"[{path_file}]: All batches loaded successfully...")
    
    def query_with_embeddings(self, embeddings, top_k):
        
        return self.collection.query(query_embeddings=embeddings,
                                      n_results=top_k)
    
    def query_with_text(self, queries, top_k, use_rerank=False):
        
        #compute embeddings
        
        embeddings_tensor = self.embedder.compute_embeddings(queries)
        embeddings_list = embeddings_tensor.tolist()
        


        ###TODO
        ## Make reranking work using the same response schems
        retrived_docs = self.query_with_embeddings(embeddings_list, top_k)
        #query augmentation coming ...

        #reranking
        if (use_rerank and self.reranker):
            
            print(type(retrived_docs))

            most_relevant_docs = self.reranker.get_most_relevant_chunks(queries[0], retrived_docs, top_k)
            return most_relevant_docs
        
        return retrived_docs
    
    
    def get(self, ids, where, limit):
        pass
    
    def reset_collection(self):
        pass
    
    
    
    