import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
import re


# def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
#     """Split text into chunks with overlap using simple whitespace split."""
#     tokens = text.split()
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = min(start + chunk_size, len(tokens))
#         chunk_str = " ".join(tokens[start:end])
#         chunks.append(chunk_str)
#         if end == len(tokens):
#             break
#         start += (chunk_size - overlap)
#     return chunks

def chunk_text_by_headings(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """
    Hybrid approach: 
    1) Split text into paragraphs by blank lines, 
    2) For large paragraphs, apply a token-based sliding window to further break them up.
    """
    # 1. Split on double newlines to identify paragraphs.
    paragraphs = text.strip().split("\n\n")
    
    final_chunks = []
    
    for paragraph in paragraphs:
        # Clean up extra whitespace
        paragraph = paragraph.strip()
        if not paragraph:
            continue  # skip empty paragraphs
        
        # Split paragraph into tokens
        tokens = paragraph.split()
        
        # 2. If paragraph is already small enough, take it as a single chunk
        if len(tokens) <= chunk_size:
            final_chunks.append(paragraph)
        
        # 3. Otherwise, break it into multiple sub-chunks, each of size chunk_size with overlap
        else:
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_str = " ".join(tokens[start:end])
                final_chunks.append(chunk_str)
                
                if end == len(tokens):
                    break
                # Move by (chunk_size - overlap)
                start += (chunk_size - overlap)
    
    return final_chunks

def chunk_text_by_headings(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32
) -> list:
    """
    Splits the entire text into sections from one heading (# ...) to the next.
    If a section (heading + subsequent text) exceeds chunk_size tokens, 
    we further split it into smaller pieces.
    """
    # Regex to match lines that start with one or more # characters
    # This will capture headings like "# Heading", "## Heading", etc.
    heading_pattern = re.compile(r'^(#+\s.*)', re.MULTILINE)
    
    # Split on headings but keep the headings themselves as separate tokens.
    # re.split with capturing parentheses retains the delimiters (the headings).
    parts = re.split(heading_pattern, text)
    
    # "parts" will look like: [pre_text, heading1, text_for_heading1, heading2, text_for_heading2, ...]
    # There's often some pre_text at the start (parts[0]) that may be empty or disclaimers, etc.
    
    chunks = []
    
    # We'll iterate in pairs: (heading, content)
    # Note: parts[0] may be text before the first heading. We'll handle that carefully.
    # For example, we can do something like:
    
    # Start from index 0 (potential text before first heading)
    # If it's not empty, keep that as a "no heading" chunk
    if parts[0].strip():
        # If you want to treat text prior to the first heading as a chunk:
        chunks.extend(_split_into_subchunks(parts[0], chunk_size, overlap))
    
    # Then from index 1, we step in increments of 2
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()          # e.g. "# C.8.7.4 Factors..."
        content = ""
        
        # The next element in parts is presumably the text after the heading
        if i+1 < len(parts):
            content = parts[i+1]
        
        # Combine heading + content so we have one big block for that section
        section_text = heading + "\n" + content
        
        # Then do final sub-chunking if it’s too long
        section_chunks = _split_into_subchunks(section_text, chunk_size, overlap)
        chunks.extend(section_chunks)
    
    return chunks

def _split_into_subchunks(text: str, chunk_size: int, overlap: int) -> list:
    """
    Helper function that splits a large block of text into smaller
    sliding windows of `chunk_size` tokens, with `overlap`.
    If text <= chunk_size, returns just one chunk.
    """
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text.strip()]
    
    subchunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_str = " ".join(tokens[start:end])
        subchunks.append(chunk_str)
        if end == len(tokens):
            break
        start += (chunk_size - overlap)
    return subchunks



def process_chunks_in_batches(chunks: List[str], model, batch_size: int = 8) -> np.ndarray:
    """Process chunks in batches to prevent memory overload."""
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"    - Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        batch_embeddings = model.encode(batch)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

def process_markdown_files(
    folder_path: str,
    model,
    chunk_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Process all markdown files in a folder using late chunking strategy with batch processing.
    Returns:
        - document_embeddings: { filename: np.ndarray }
        - chunk_embeddings: { filename: np.ndarray }
        - chunk_texts: { filename: list of strings (the actual chunk contents) }
    """
    document_embeddings = {}
    chunk_embeddings = {}
    chunk_texts = {} 
    
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
        
        # 2. Perform late chunking
        print(" - Performing late chunking...")
        chunks = chunk_text_by_headings(text, chunk_size=chunk_size, overlap=overlap)
        chunk_texts[filename] = chunks  # store the chunks for later retrieval
        print(f" - Created {len(chunks)} chunks")
        
        # 3. Compute embeddings for chunks in batches
        print(" - Computing chunk embeddings in batches...")
        chunk_embs = process_chunks_in_batches(chunks, model, batch_size)
        chunk_embeddings[filename] = chunk_embs
        print(f" - Chunk embeddings shape: {chunk_embs.shape}")
        
        # 4. Optional: Some quick similarity stats
        if len(chunks) > 1:
            sample_size = min(10, len(chunks))
            sample_embeddings = chunk_embs[:sample_size]
            chunk_similarities = np.mean(sample_embeddings @ sample_embeddings.T) * 100
            print(f" - Average chunk similarity (from sample): {chunk_similarities:.2f}%")
    
    return document_embeddings, chunk_embeddings, chunk_texts

def store_embeddings_locally(
    doc_embeddings: Dict[str, np.ndarray],
    chunk_embeddings: Dict[str, np.ndarray],
    chunk_texts: Dict[str, List[str]],
    output_folder: str = "embeddings_by_heading"
):
    """
    Save the document embeddings (np.ndarrays), chunk embeddings, 
    and chunk texts to disk so we can load/query later.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Save document embeddings
    for filename, emb in doc_embeddings.items():
        np.save(os.path.join(output_folder, f"{filename}_doc_emb.npy"), emb)
    
    # 2. Save chunk embeddings + texts
    for filename, emb_array in chunk_embeddings.items():
        np.save(os.path.join(output_folder, f"{filename}_chunk_embs.npy"), emb_array)
        
        # Save the chunk texts in JSON
        chunk_list = chunk_texts[filename]
        with open(os.path.join(output_folder, f"{filename}_chunk_texts.json"), 'w', encoding='utf-8') as f:
            json.dump(chunk_list, f, ensure_ascii=False, indent=2)



def main():
    # Initialize model
    print("Initializing Qwen model...")
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
    model.max_seq_length = 8192
    
    # Set parameters
    folder_path = "data"    
    chunk_size = 256        
    overlap = 32            
    batch_size = 8          
    
    # Process markdown files (returns doc embeddings, chunk embeddings, and chunk texts)
    print(f"\nProcessing markdown files from {folder_path}")
    doc_embeddings, chunk_embeddings, chunk_texts = process_markdown_files(
        folder_path, 
        model,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size
    )
    
    # Store them locally
    store_embeddings_locally(doc_embeddings, chunk_embeddings, chunk_texts, output_folder="embeddings_by_heading")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {len(doc_embeddings)} files and stored embeddings in 'embeddings' folder.")

if __name__ == "__main__":
    main()
