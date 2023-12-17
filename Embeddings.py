import openai
import chromadb
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("api")

# Initialize ChromaDB Client
chroma_client = chromadb.Client()

# Create a collection in ChromaDB
collection_name = "articles_embeddings"
collection = chroma_client.create_collection(name=collection_name)

# Function to process CSV files in a directory and add them to ChromaDB
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            country_code = filename[:2]  # Assuming the first two letters are the country code
            process_csv(file_path, country_code)

# Function to get embeddings using OpenAI's API
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def process_csv(file_path, country_code):
    df = pd.read_csv(file_path)

    # Combine the necessary text fields if needed
    # Example: df['combined_text'] = df['Summary'] + " " + df['Text']
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns

    # Obtain embeddings and convert to list
    df['embedding'] = df['combined_text'].apply(lambda text: get_embedding(text))
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    # Iterate over the dataframe and add each row to ChromaDB
    for index, row in df.iterrows():
        collection.add(
            embeddings=[row['embedding'].tolist()],  # Convert numpy array to list
            documents=[row['combined_text']],
            metadatas=[{
                'article_id': row['article_id'],
                'article_title': row['article_title'],
                'publisher': row['publisher'],
                'year': row['year'],
                'Domestic': row['Domestic'],
                'country_code': country_code
            }],
            ids=[row['article_id']]
        )


# Path to the directory containing CSV files
directory_path = '/Users/kev/Downloads/now'

# Process all CSV files in the directory
process_directory(directory_path)
