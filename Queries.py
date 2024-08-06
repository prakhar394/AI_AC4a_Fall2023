'''
Code for peace classification of countries from vector database
'''
import os
import pandas as pd
import getpass
from collections import Counter
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# Initialize the vector database
os.environ["OPENAI_API_KEY"] = getpass.getpass()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'db'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


def process_query_csv(file_path, vectordb, file_country_code):
    print(f'Processing query CSV for {file_country_code}...')
    df = pd.read_csv(file_path, nrows=1000)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize'].str[:1000]

    country_similarity = Counter()
    peaceful_count = 0
    total_rows = 0
    overall_peaceful = None  # Initialize overall_peaceful

    # Perform similarity search for each row in the DataFrame
    for index, row in df.iterrows():
        query_text = row['combined_text']
        # Get the most similar documents
        similar_docs = vectordb.similarity_search(query_text)

        # Filter out documents where the country code matches the file's country code
        filtered_docs = [doc for doc in similar_docs if doc.metadata.get('country_code', 'Unknown') != file_country_code]

        # Proceed if there are any documents left after filtering
        if filtered_docs:
            total_rows += 1
            most_similar_doc = filtered_docs[0]  # The top result after filtering
            country_code = most_similar_doc.metadata.get('country_code', 'Unknown')
            is_peaceful = most_similar_doc.metadata.get('peaceful', False)
            country_similarity[country_code] += 1
            peaceful_count += is_peaceful

    # Determine the overall result for the file
    if total_rows > 0:
        most_common_country, _ = country_similarity.most_common(1)[0]
        overall_peaceful = peaceful_count / total_rows > 0.5  # Majority vote
        peace_percentage = 100*peaceful_count / total_rows
        print(f"Overall, the country is {'peaceful' if overall_peaceful else 'not peaceful'}")
        print(f"The country is most similar to {most_common_country}")
    else:
        print(f"No conclusive result for {file_country_code}. Unable to determine if the country is peaceful or not.")

    return overall_peaceful, peace_percentage


def process_directory(directory_path, vectordb):
    country_results = Counter()
    country_peace_percentages = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            file_country_code = filename[:2]  # Assuming the first two letters are the country code

            # Process each CSV file
            is_peaceful, peace_percentage = process_query_csv(file_path, vectordb, file_country_code)
            country_results[file_country_code] = is_peaceful
            country_peace_percentages[file_country_code] = peace_percentage

    # Print the results for each country
    for country_code, is_peaceful in country_results.items():
        print(f"{country_code}: {'Peaceful' if is_peaceful else 'Not Peaceful'}")


    # Print peace percentages for each country
    print("\nPeace Percentages:")
    for country_code, peace_percentage in country_peace_percentages.items():
        print(f"{country_code}: {peace_percentage:.2f}%")

    return country_results


# Example usage
path = os.environ.get("directory")
process_directory(path, vectordb)
