from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from embedding_model import Embedder
import chromadb
import torch
import hashlib
import numpy as np

import math



def generate_sha256_hash_from_text(text):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))

    return sha256_hash.hexdigest()

class PdfChunksLoader_ChromaDB():
    def __init__(self, collection, embedder, text_splitter=None):
        
        
        
        self.collection = collection
        self.embedder = embedder
        self.id = 0
        self.text_splitter = text_splitter if text_splitter else RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                                           chunk_overlap=100,
                                                                           separators=["\n", "\t", ".", ",", " ", ""],)
        
        
    def _extract_pdf_chunks(self, path):
        
        loader = PyPDFLoader(path)
        
        chunks = loader.load_and_split(text_splitter=self.text_splitter)
        
        return chunks
    
    def _get_chunk_id(self, chunk):
        
        return "chunkID_" + generate_sha256_hash_from_text(chunk.page_content)
        
    def filter_existing_docs(self, docs_ids_map):
        
        
        ids_computed = list(docs_ids_map.keys())
        
            
        existing_chunks_ids = self.collection.get(ids=ids_computed)["ids"]
        
        
        def extract_only_new_docs(keyval_tuple):
            key, value = keyval_tuple
            
            return (key not in existing_chunks_ids)
            
        filtered_docs_map = dict(filter(extract_only_new_docs,  docs_ids_map.items()))
        
        return filtered_docs_map
        
        
    
    def populate(self, documents):
        ##TODO: add batch size for computing embeddings
        
        ### try to add one by one to avoid redundant computing of embedds
        
        
        #check filter ids
        
        ids_computed = [self._get_chunk_id(chunk) for chunk in documents]
        
        docs_id_map = {uri_id : doc for uri_id, doc in zip(ids_computed, documents)}
        
        filtered_docs_id_map = self.filter_existing_docs(docs_id_map)
        
        if (filtered_docs_id_map):
            self.collection.add(
                documents=[chunk.page_content for chunk in filtered_docs_id_map.values()],

                metadatas = [chunk.metadata for chunk in filtered_docs_id_map.values()],

                embeddings = self.embedder.compute_embeddings([chunk.page_content for chunk in filtered_docs_id_map.values()]).tolist(),
                ids = [doc_id for doc_id in filtered_docs_id_map.keys()]

                )
        else:
            print("Documents already exist...")

class CrossEncoderReRanker():
    def __init__(self, cross_encoder):
        self.cross_encoder = cross_encoder
        
    def _compute_scores(self, query, documents):
        
        #pairs = [[query, doc.page_content] for doc in documents]
        pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        return scores
    
    def get_most_relevant_chunks(self, query, documents, n):
        
        scores = self._compute_scores(query, documents)
        sorted_indices = np.argsort(scores)[::-1]
        
        relevant_docs = [documents[i] for i in sorted_indices[:n]]
        
        return relevant_docs
        
        
    
    