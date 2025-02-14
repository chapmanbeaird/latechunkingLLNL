import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_embeddings_local(filename_base: str, embeddings_folder: str = "embeddings_by_heading"):
    """
    Given a base filename (e.g. 'readme.md'), load its chunk embeddings and chunk texts.
    """
    emb_path = os.path.join(embeddings_folder, f"{filename_base}_chunk_embs.npy")
    texts_path = os.path.join(embeddings_folder, f"{filename_base}_chunk_texts.json")

    if not os.path.exists(emb_path) or not os.path.exists(texts_path):
        raise FileNotFoundError(f"Embedding or text file not found for {filename_base}")

    chunk_embs = np.load(emb_path)
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        chunk_texts = json.load(f)
    
    return chunk_embs, chunk_texts

def cosine_similarity_search(query: str, model, chunk_embs: np.ndarray, chunk_texts: list, top_k: int = 3):
    """
    Search local chunk embeddings for the chunks most similar to the query.
    Returns a list of (chunk_text, similarity_score).
    """
    # 1. Encode the query
    query_emb = model.encode(query)
    
    # 2. Compute dot product
    dot_products = chunk_embs @ query_emb  
    # 3. Normalize for cosine similarity
    chunk_norms = np.linalg.norm(chunk_embs, axis=1)
    query_norm = np.linalg.norm(query_emb)
    similarities = dot_products / (chunk_norms * query_norm + 1e-10)
    
    # 4. Sort and pick top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(chunk_texts[i], float(similarities[i])) for i in top_indices]
    return results

def main():
    # 1. Load the same model used for embedding
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
    
    # 2. Pick which file you want to query against (e.g. 'readme.md')
    filename_base = "591772.md" 
    
    # 3. Load embeddings/text for that file
    chunk_embs, chunk_texts = load_embeddings_local(filename_base, embeddings_folder="embeddings_by_heading")
    
    # 4. Ask your query
    # query_str = "What factors affect BZA air sampling?"
    query_str = input("What is your query?")
    
    # 5. Retrieve top matches
    top_results = cosine_similarity_search(query_str, model, chunk_embs, chunk_texts, top_k=3)
    
    # 6. Print results
    for rank, (chunk_text, score) in enumerate(top_results, start=1):
        print(f"\n=== Result #{rank} (score: {score:.4f}) ===")
        print(chunk_text)

if __name__ == "__main__":
    main()
