import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love programming in Python",
    "Machine learning is fascinating",
    "Cats are sleeping on the sofa",
    "Artificial intelligence is changing the world",
    "I enjoy writing code",
    "The weather is very sunny today",
    "Dogs are loyal animals",
    "Deep learning uses neural networks",
    "Pizza tastes amazing",
    "Software engineering is creative"
]

embeddings = model.encode(sentences) #convert sentences into embeddings
embeddings = np.array(embeddings).astype("float32") #convert to flaot32 numpy array

dimension = embeddings.shape[1] #no of columns. 384 for this model
index = faiss.IndexFlatL2(dimension) #uses euclidean distance
index.add(embeddings)

print(f"Index built. Total vectors stored: {index.ntotal}")

query="I like coding"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

#get top 3most similar 

distances, indices = index.search(query_embedding, k=3)

print(f"\nQuery: {query}")
print(f"\nTop 3 results:")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {rank+1}. {sentences[idx]}  (distance: {dist:.4f})")