import numpy as np
from sentence_transformers import SentenceTransformer

def compute_embeddings(texts):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings