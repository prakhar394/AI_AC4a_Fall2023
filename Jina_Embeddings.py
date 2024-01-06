import os
import pandas as pd
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

path = os.environ.get("directory")
db_path = os.environ.get("db")
api = os.environ.get("api")

# Initialize LangChain's Jina Embeddings
jina_embeddings = JinaEmbeddings(jina_auth_token=api, model_name='jina-embeddings-v2-base-en')

# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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
    df = pd.read_csv(file_path,nrows=10)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns

    all_texts = []
    # all_metadatas = []

    for index, row in df.iterrows():
        # Split text into smaller segments
        segments = text_splitter.split_text(row['combined_text'])
        all_texts.extend(segments)

        # Create metadata for each segment
        metadata = [{
            'article_id': row['article_id'],
            'article_title': row['article_title'],
            'publisher': row['publisher'],
            'year': row['year'],
            'Domestic': row['Domestic'],
            'country_code': row['country_code']  # Assuming country code is a column in your CSV
        } for _ in segments]

        # all_metadatas.extend(metadata)

    # Initialize ChromaDB and store documents with embeddings
    collection = Chroma.from_documents(all_texts, jina_embeddings)

    print('Processing complete.')

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
