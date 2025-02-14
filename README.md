This project uses the Sentence Transformers library to process text documents, generate embeddings, and perform similarity-based queries. Below is an overview of the key files and what they do:

Files and Their Functions

query_embeddings.py

Purpose: Loads precomputed embeddings from local files and performs cosine similarity searches on them.

Key Functions:

load_embeddings_local: Loads chunk embeddings and text chunks from .npy and .json files.

cosine_similarity_search: Computes similarity scores between a query embedding and the loaded chunk embeddings.

main: Runs the query process, prompting the user for a query and displaying top matching text chunks.

test.py

Purpose: A simple script to test the Sentence Transformer model by encoding sample queries and documents, then computing similarity scores.

Key Components:

Loads the Qwen model.

Encodes provided queries and documents.

Prints similarity scores between the encoded queries and documents.

latechunking.py

Purpose: Handles text chunking and embedding generation for markdown files.

Key Functions:

chunk_text_by_headings: Splits text into sections based on headings, further chunking large sections.

_split_into_subchunks: Helper for breaking large text blocks into smaller overlapping chunks.

process_chunks_in_batches: Processes text chunks in batches for embedding generation.

process_markdown_files: Processes all markdown files in a folder, generating document and chunk embeddings.

store_embeddings_locally: Saves embeddings and text chunks to local storage.

Main Function: Initializes the model, sets parameters, processes markdown files, and stores the embeddings locally.

Data folder

Purpose: Markdown files used as input for text chunking and embedding.

How the Project Works

latechunking.py processes markdown files by splitting them into manageable text chunks and generating embeddings.

These embeddings are stored locally for future queries.

query_embeddings.py loads these stored embeddings and allows the user to query the text using cosine similarity.

test.py is used to quickly verify model functionality and embeddings generation.

This structure ensures efficient handling of large text documents and enables similarity-based text retrieval with ease.
