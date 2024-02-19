import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
load_dotenv()

path = os.environ.get("directory")
# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'db'
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def load_peaceful_countries_data():
    # Path to your CSV file
    csv_file_path = path+'/peaceful/peaceful_countries.csv'

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return {}

    # Convert the DataFrame to a dictionary
    peaceful_countries = dict(zip(df['country_code'], df['peaceful']))
    return peaceful_countries

# Load the peaceful countries data
peaceful_countries = load_peaceful_countries_data()

# Function to process CSV files in a directory
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            country_code = filename[:2]  # Assuming the first two letters are the country code
            is_peaceful = peaceful_countries.get(country_code, False)
            print("Processing " + country_code + "!")
            process_csv(file_path, country_code, is_peaceful, vectordb)

    # Function to process a CSV file and add documents to ChromaDB
def process_csv(file_path, country_code, is_peaceful, vectordb):
    print('Initializing...')
    # Count the total number of rows in the CSV file (excluding the header row)
    total_rows = sum(1 for _ in open(file_path)) - 1
    # Calculate the number of rows to skip
    skip_rows = np.sort(np.random.choice(range(1, total_rows + 1), size=(total_rows - 2300), replace=False))
    # Read 2300 random rows from the CSV file
    df = pd.read_csv(file_path, skiprows=skip_rows, nrows=2300)
    df['combined_text'] = df['article_text_Ngram'].str[:1000]  # Replace with your text columns
    df['document'] = df['combined_text'].apply(lambda x: Document(page_content=x, metadata={'peaceful': is_peaceful, 'country_code': country_code}))
    print('Document retrieved!')
    documents = df['document'].tolist()
    texts = text_splitter.split_documents(documents)
    print('Text split!')
    # Stores embeddings into chromadb database
    vectordb.add_documents(texts)

    print('Processing complete.')

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
process_directory(directory_path)
#process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
