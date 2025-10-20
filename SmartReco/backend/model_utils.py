# backend/model_utils.py
"""
Utility functions for embeddings, similarity, and text preprocessing.
Part of the SmartReco backend system.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load a lightweight, high-quality embedding model
# (This can be replaced with OpenAI embeddings or any custom model)
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def clean_text(text: str) -> str:
    """
    Basic preprocessing for text data.
    Removes special characters, extra spaces, and lowercases.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def get_text_embedding(text: str):
    """
    Generates sentence embeddings for a given text using SentenceTransformer.
    """
    text = clean_text(text)
    return embedder.encode([text])[0]


def get_batch_embeddings(texts):
    """
    Generates embeddings for a batch of text items.
    """
    texts = [clean_text(t) for t in texts]
    return embedder.encode(texts, convert_to_tensor=True)


def calculate_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two embeddings.
    """
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().detach().numpy()
    return cosine_similarity([vec1], [vec2])[0][0]


def top_k_similar(query_embedding, all_embeddings, k=5):
    """
    Returns the top-k most similar items based on cosine similarity.
    """
    if isinstance(all_embeddings, torch.Tensor):
        all_embeddings = all_embeddings.cpu().detach().numpy()
    sims = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return top_indices, sims[top_indices]
