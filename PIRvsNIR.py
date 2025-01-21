import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize embeddings and vector databases
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
peacedb = Chroma(persist_directory="db", embedding_function=embedding_function)

# Define PIR and NIR definitions and their embeddings
PIR = "INTERGROUP TOLERANCE, RESPECT, KINDNESS, HELP OR SUPPORT"
NIR = "INTERGROUP INTOLERANCE, DISRESPECT, AGGRESSION, OBSTRUCTION, OR HINDRANCE"

pir_embedding = embedding_function.embed_query(PIR)
nir_embedding = embedding_function.embed_query(NIR)

# Function to find the most similar articles from the database
def find_most_similar_articles(target_embedding, db, top_k=5, exclude_metadata=None):
    results = db.similarity_search_by_vector(target_embedding, k=top_k * 20)  # Retrieve extra to filter later
    similar_articles = []

    for result in results:
        if exclude_metadata:
            # Check if any metadata keys to exclude match
            if any(result.metadata.get(key) == value for key, value in exclude_metadata.items()):
                continue

        similar_articles.append({
            "content": result.page_content,
            "metadata": result.metadata,
            "similarity_score": result.metadata.get("similarity_score", None),
        })

        if len(similar_articles) == top_k:  # Stop when we have enough valid articles
            break

    return similar_articles

# Retrieve the top 5 most similar articles for PIR and NIR
top_5_pir = find_most_similar_articles(pir_embedding, peacedb, top_k=7)
top_5_nir = find_most_similar_articles(nir_embedding, peacedb, top_k=7, exclude_metadata={"country_code": "KE"})

# Print the results
print("\nTop 5 Most Similar Articles for PIR:")
for i, article in enumerate(top_5_pir, start=1):
    print(f"Article {i}:")
    print(f"Content: {article['content']}...")
    print(f"Metadata: {article['metadata']}")
    print()

print("\nTop 5 Most Similar Articles for NIR (Excluding KE):")
for i, article in enumerate(top_5_nir, start=1):
    print(f"Article {i}:")
    print(f"Content: {article['content']}...")
    print(f"Metadata: {article['metadata']}")
    print()
