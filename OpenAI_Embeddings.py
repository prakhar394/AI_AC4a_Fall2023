import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
import openai


load_dotenv()

path = os.environ.get("directory")
llm = ChatOpenAI(temperature=0, model_name='text-embedding-ada-002')
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
    df = pd.read_csv(file_path)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize']  # Replace with your text columns

    db = Chroma()

    for index, row in df.iterrows():
        # Split text into smaller segments
        articles_doc = [Document(page_content=row['combined_text'])]
        texts = text_splitter.split_documents(articles_doc)

        # Stores embeddings into chromadb database
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)

    print('Processing complete.')

    #query = "Where is Tanzania"
    #docs = db.similarity_search(query)
    # print results
    #print(docs[0].page_content)

# Path to the directory containing CSV files
directory_path = path

# Process all CSV files in the directory
# process_directory(directory_path)
process_csv(directory_path+'/TZ_domestic_Ngram_stopword_lematize.csv', 'TZ')
