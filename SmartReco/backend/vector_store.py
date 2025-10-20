# vector_store.py
# Handles embeddings creation, FAISS index building, and semantic search

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, fast embedding model
        self.index = None
        self.product_data = None
        self.embeddings = None

    def load_data(self, path: str):
        """Load product dataset"""
        self.product_data = pd.read_csv(path)
        self.product_data.fillna("", inplace=True)
        print(f"✅ Loaded {len(self.product_data)} products.")

    def create_embeddings(self):
        """Generate text embeddings using product title + description"""
        texts = (
            self.product_data["title"].astype(str)
            + " " +
            self.product_data["description"].astype(str)
        ).tolist()

        print("🔄 Generating embeddings...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print("✅ Embeddings generated successfully.")
        return self.embeddings

    def build_faiss_index(self, embeddings):
        """Build FAISS vector index"""
        d = embeddings.shape[1]  # embedding dimension
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        print(f"✅ FAISS index built with {embeddings.shape[0]} entries.")

    def search(self, query: str, top_k: int = 5):
        """Perform semantic search for a query"""
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            product = self.product_data.iloc[idx]
            results.append({
                "rank": i + 1,
                "title": product["title"],
                "brand": product["brand"],
                "description": product["description"],
                "price": product["price"],
                "categories": product["categories"],
                "image": product["images"],
                "score": float(distances[0][i])
            })
        return results
