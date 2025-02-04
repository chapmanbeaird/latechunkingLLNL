import os
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=256, overlap=32):
    """Split text into chunks with overlap using simple whitespace split."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_str = " ".join(tokens[start:end])
        chunks.append(chunk_str)
        if end == len(tokens):
            break
        start += (chunk_size - overlap)
    return chunks

def process_markdown_files(folder_path, model, chunk_size=256, overlap=32):
    """Process all markdown files in a folder using late chunking strategy."""
    document_embeddings = {}
    chunk_embeddings = {}
    
    # Iterate through markdown files
    for filename in os.listdir(folder_path):
        if not filename.endswith('.md'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing file: {filename}")
        
        # Read the markdown file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 1. First compute full document embedding
        print(" - Computing full document embedding...")
        full_doc_embedding = model.encode([text])[0]
        document_embeddings[filename] = full_doc_embedding
        print(f" - Full document embedding shape: {full_doc_embedding.shape}")
        
        # 2. Then perform late chunking
        print(" - Performing late chunking...")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f" - Created {len(chunks)} chunks")
        
        # 3. Compute embeddings for chunks
        print(" - Computing chunk embeddings...")
        chunk_embs = model.encode(chunks)
        chunk_embeddings[filename] = chunk_embs
        print(f" - Chunk embeddings shape: {chunk_embs.shape}")
        
        # 4. Optional: Compute and print some statistics
        if len(chunks) > 1:
            chunk_similarities = np.mean(chunk_embs @ chunk_embs.T) * 100
            print(f" - Average chunk similarity: {chunk_similarities:.2f}%")
    
    return document_embeddings, chunk_embeddings

def main():
    # Initialize model
    print("Initializing Qwen model...")
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
    model.max_seq_length = 8192
    
    # Set parameters
    folder_path = "data"  # Folder containing markdown files
    chunk_size = 256      # Number of tokens per chunk
    overlap = 32          # Number of overlapping tokens between chunks
    
    # Process all markdown files
    print(f"\nProcessing markdown files from {folder_path}")
    print(f"Using chunk size: {chunk_size}, overlap: {overlap}")
    
    doc_embeddings, chunk_embeddings = process_markdown_files(
        folder_path, 
        model,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {len(doc_embeddings)} files")
    
    # Optional: Example of how to use the embeddings
    print("\nExample similarity calculation between documents:")
    filenames = list(doc_embeddings.keys())
    if len(filenames) >= 2:
        doc1, doc2 = filenames[0], filenames[1]
        similarity = np.dot(doc_embeddings[doc1], doc_embeddings[doc2]) * 100
        print(f"Similarity between {doc1} and {doc2}: {similarity:.2f}%")

if __name__ == "__main__":
    main()