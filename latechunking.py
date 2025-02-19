from transformers import AutoModel
from transformers import AutoTokenizer
import numpy as np
import torch


# Initialize the model
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)


# Read your markdown file
with open('./data/591772.md', 'r') as file:  
    markdown_text = file.read()

# Define the chunking function (same as their example)
def chunk_by_sentences(input_text: str, tokenizer: callable):
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations

# Define the late chunking function
def late_chunking(model_output, span_annotation, max_length=None):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)
    return outputs

# Process your markdown text
# 1. Get chunks and span annotations
chunks, span_annotations = chunk_by_sentences(markdown_text, tokenizer)

# 2. Get embeddings using both methods
# Traditional chunking
traditional_embeddings = model.encode(chunks)

# Late chunking
inputs = tokenizer(markdown_text, return_tensors='pt')
model_output = model(**inputs)
late_chunked_embeddings = late_chunking(model_output, [span_annotations])[0]

# Print some information about the chunks
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
# cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# # Let's compare chunks with a relevant query from your markdown content
# # You can change this query to whatever term you want to search for
# query = "machine learning"  # or any other relevant term from your markdown
# query_embedding = model.encode(query)

# print(f"\nComparing chunks with query: '{query}'")
# print("-" * 50)

# for i, (chunk, late_embedding, trad_embedding) in enumerate(zip(chunks, late_chunked_embeddings, traditional_embeddings)):
#     # Calculate similarities
#     late_sim = cos_sim(query_embedding, late_embedding)
#     trad_sim = cos_sim(query_embedding, trad_embedding)
    
#     # Print results
#     print(f"\nChunk {i} (first 100 chars): {chunk[:100]}...")
#     print(f"Late chunking similarity: {late_sim:.4f}")
#     print(f"Traditional similarity: {trad_sim:.4f}")
#     print("-" * 50)