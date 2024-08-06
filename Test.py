'''
Code for printing sample articles from Nigeria and New Zealand from vector database
'''

import os
import pandas as pd
import random
from time import sleep
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# print('status')
load_dotenv()
# Initialize the vector database
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'db'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
path = os.environ.get("directory")


def load_peaceful_countries_data():
    csv_file_path = path + '/peaceful/peaceful_countries.csv'

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


def process_query_csv(file_path, vectordb, file_country_code, nrows):
    # Determine the total number of rows in the file
    total_rows = sum(1 for _ in open(file_path, 'r')) - 1  # Subtract 1 for the header row

    # Randomly select a starting point
    start_row = random.randint(0, total_rows - nrows)

    # Read nrows rows from the random starting point
    df = pd.read_csv(file_path, skiprows=range(1, start_row + 1), nrows=nrows)

    df['combined_text'] = df['article_text_Ngram'].str[:1000]
    # df['combined_text'] = df['article_text_Ngram']

    selected_articles = []

    for index, row in df.iterrows():
        try:
            query_text = row['combined_text']

            # Ensure query_text is a string
            if not isinstance(query_text, str):
                raise ValueError(f"Non-string query_text at row {index}: {query_text}")

            # print(f"Processing row {index} with query_text: {query_text[:100]}")  # Print the first 100 characters
            k_val = 10
            # Get the most similar documents
            similar_docs = vectordb.similarity_search(query_text, k=k_val)

            # Filter out documents where the country code matches the file's country code
            filtered_docs = [doc for doc in similar_docs if
                             doc.metadata.get('country_code', 'Unknown') != file_country_code]

            while not filtered_docs:
                k_val *= 2
                similar_docs = vectordb.similarity_search(query_text, k=k_val)

                # Filter out documents where the country code matches the file's country code
                filtered_docs = [doc for doc in similar_docs if
                                 doc.metadata.get('country_code', 'Unknown') != file_country_code]

            # Proceed if there are any documents left after filtering
            if filtered_docs:
                # Always take the first document from the list
                most_similar_doc = filtered_docs[0]
                country_code = most_similar_doc.metadata.get('country_code', 'Unknown')
                is_peaceful = most_similar_doc.metadata.get('peaceful', False)
                peaceful_flag = 1 if is_peaceful else 0
                selected_articles.append((row['article_text_Ngram'][:2000], most_similar_doc.page_content[:2000],
                                          peaceful_flag, country_code))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            sleep(30)
            continue

    print(f"Processed {len(selected_articles)} rows from file {file_path}")
    return selected_articles


def process_directory(directory_path, vectordb):
    nrows = 1024
    selected_articles_nigeria = []
    selected_articles_new_zealand = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            file_country_code = filename[:2]  # Assuming the first two letters are the country code
            if file_country_code == "NG":  # Nigeria
                selected_articles_nigeria.extend(process_query_csv(file_path, vectordb, file_country_code, nrows))
            elif file_country_code == "NZ":  # New Zealand
                selected_articles_new_zealand.extend(process_query_csv(file_path, vectordb, file_country_code, nrows))

    # Save selected articles to CSV files
    save_to_csv('nigeria_articles.csv', selected_articles_nigeria)
    save_to_csv('new_zealand_articles.csv', selected_articles_new_zealand)


def save_to_csv(file_name, articles):
    df = pd.DataFrame(articles, columns=['article_text', 'most_similar_doc_text', 'peaceful_flag', 'country_code'])
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    path = os.environ.get("directory")
    path += "/train"
    process_directory(path, vectordb)
