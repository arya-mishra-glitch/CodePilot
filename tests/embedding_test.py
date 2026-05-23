from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLm-L6-v2")

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

embeddings = model.encode(sentences)

query = "I like cats"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]

best_index = similarities.argmax()

print(f"\nQuery: {query}")
print(f"Most similar sentence: {sentences[best_index]}")
print(f"Similarity Score: {similarities[best_index]:.4f}")

for i, (sentence, score) in enumerate(zip(sentences, similarities)):
    print(f"{score:.4f}  {sentence}")