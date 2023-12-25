import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

path = os.environ.get("directory")
api_key = os.environ.get("OPENAI_API_KEY")
# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
persist_directory = 'db'
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
    df = pd.read_csv(file_path)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns
    df['document'] = df['combined_text'].apply(lambda x: Document(page_content=x))
    print('Document retrieved!')
    documents = df['document'].tolist()
    texts = text_splitter.split_documents(documents)
    print('Text split!')
    # Stores embeddings into chromadb database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)

    print('Processing complete.')
    query = "golf"
    docs = vectordb.similarity_search(query)
    # print results
    print(docs[0].page_content)

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
