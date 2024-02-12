import os
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Initialize the vector database
persist_directory = 'sorteddb'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


def get_peace_percentage(peace_scores):
    # Convert scores to a percentage, assuming 1 is 0%, 2 is 33%, 3 is 67%, and 4 is 100%
    percentages = {
        1: 0,
        2: 33,
        3: 67,
        4: 100
    }
    return [percentages.get(score, 0) for score in peace_scores]


def classify_country(average_percentage):
    # Define thresholds for classification
    if average_percentage < 25:
        return 'Nonpeaceful'
    elif average_percentage < 50:
        return 'Slightly Nonpeaceful'
    elif average_percentage < 75:
        return 'Slightly Peaceful'
    else:
        return 'Peaceful'


def process_query_csv(file_path, vectordb, file_country_code):
    print(f'Processing query CSV for {file_country_code}...')
    df = pd.read_csv(file_path, nrows=1000)
    df['combined_text'] = df['article_text_Ngram_stopword_lemmatize'].str[:1000]

    peace_scores = []

    # Perform similarity search for each row in the DataFrame
    for index, row in df.iterrows():
        query_text = row['combined_text']
        similar_docs = vectordb.similarity_search(query_text)
        filtered_docs = [doc for doc in similar_docs if
                         doc.metadata.get('country_code', 'Unknown') != file_country_code]

        for doc in filtered_docs:
            peace_scores.append(doc.metadata.get('peace_score', 0))  # Assuming 0 if unknown

    # Calculate the average peace percentage for the country
    if peace_scores:
        peace_percentages = get_peace_percentage(peace_scores)
        average_percentage = sum(peace_percentages) / len(peace_percentages)
        country_classification = classify_country(average_percentage)
        print(f"{file_country_code}: {country_classification} ({average_percentage:.2f}%)")
    else:
        average_percentage = 0
        country_classification = 'Data Insufficient'
        print(f"No conclusive result for {file_country_code}. Unable to determine peace status.")

    return average_percentage, country_classification


def process_directory(directory_path, vectordb):
    country_results = Counter()
    country_peace_percentages = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            file_country_code = filename[:2]  # Assuming the first two letters are the country code

            # Process each CSV file
            average_percentage, country_classification = process_query_csv(file_path, vectordb, file_country_code)
            country_results[file_country_code] = country_classification
            country_peace_percentages[file_country_code] = average_percentage

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
