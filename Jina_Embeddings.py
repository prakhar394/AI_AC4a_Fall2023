import os
import pandas as pd
from transformers import AutoModel
import numpy as np
import chromadb
from dotenv import load_dotenv

load_dotenv()

path = os.environ.get("directory")

# Initialize Jina's embedding model
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

# Function to get embeddings using Jina's model
def get_jina_embeddings(text, max_length=2048):
    if(text):
        print(text[0])
    return model.encode(text, max_length=max_length)

# Initialize ChromaDB Client
chroma_client = chromadb.Client()

# Create a collection in ChromaDB
collection_name = "articles_embeddings"
collection = chroma_client.create_collection(name=collection_name)

# Function to process CSV files in a directory
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            country_code = filename[:2]  # Assuming the first two letters are the country code
            process_csv(file_path, country_code)


# Function to process a CSV file and add documents to ChromaDB
def process_csv(file_path, country_code):
    print("bye")
    df = pd.read_csv(file_path)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns

    # Obtain embeddings
    df['embedding'] = df['combined_text'].apply(lambda text: get_jina_embeddings(text))

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
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv('/Users/kev/Downloads/now/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')