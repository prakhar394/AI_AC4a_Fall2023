import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
path = os.environ.get("directory")

chroma_client = chromadb.Client()

embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name="text-embedding-ada-002")

# Create a collection in ChromaDB
collection_name = "articles_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name,embedding_function=embedding_function)

# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Function to process CSV files in a directory
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            country_code = filename[:2]  # Assuming the first two letters are the country code
            process_csv(file_path, country_code)

# Function to process a CSV file and add documents to ChromaDB
def process_csv(file_path, country_code):
    print('Initializing...')
    df = pd.read_csv(file_path, nrows=10)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns

    for index, row in df.iterrows():
        # Split text into smaller segments
        segments = text_splitter.split_text(row['combined_text'])

        # Generate embeddings for each segment and add to ChromaDB
        for segment in segments:
            embedding = embedding_function([segment])

            # Add segment and its embedding to ChromaDB
            collection.add(
                documents=[segment],
                embeddings=[embedding],
                metadatas=[{
                    'article_id': row['article_id'],
                    'article_title': row['article_title'],
                    'publisher': row['publisher'],
                    'year': row['year'],
                    'Domestic': row['Domestic'],
                    'country_code': country_code
                }],
                ids=[f"{row['article_id']}-{index}"]  # Example of creating a unique ID for each segment
            )

    print('Processing complete.')

    print(collection.peek())
    print(collection.count())

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
