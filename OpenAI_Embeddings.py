import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import time
import random
import openai

load_dotenv()

path = os.environ.get("directory")
api_key = os.environ.get("OPENAI_API_KEY")
# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
persist_directory = 'vectordb'
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
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']
    df['document'] = df['combined_text'].apply(lambda x: Document(page_content=x))
    documents = df['document'].tolist()
    texts = text_splitter.split_documents(documents)

    # Batch processing variables
    batch_size = 50  # Adjust this based on your rate limit and document size
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

    def exponential_backoff(retry):
        # Wait for 2^retry * 100 milliseconds
        time.sleep((2 ** retry) * 0.1 + (random.randint(0, 1000) / 1000))

    # Function to process a single batch
    def process_batch(batch_texts):
        retry = 0
        max_retries = 5  # Maximum number of retries
        while retry < max_retries:
            try:
                embeddings = OpenAIEmbeddings()
                db = Chroma.from_documents(batch_texts, embedding=embeddings, persist_directory=persist_directory)
                return True  # Successful processing of the batch
            except openai.RateLimitError:
                retry += 1
                exponential_backoff(retry)
        return False  # Failed after max retries

    # Iterate over batches
    for batch_num in range(total_batches):
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch_texts = texts[start_index:end_index]

        if not process_batch(batch_texts):
            print(f"Failed to process batch {batch_num + 1} after several retries.")
            break  # Exit the loop if a batch fails after retries

    print('Processing complete.')
    query = "golf"
    docs = db.similarity_search(query)
    # print results
    print(docs[0].page_content)

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
