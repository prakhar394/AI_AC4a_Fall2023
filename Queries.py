import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
load_dotenv()

# Initialize the vector database
persist_directory = 'db'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def process_query_csv(file_path, vectordb):
    print('Processing query CSV...')
    df = pd.read_csv(file_path)
    df['combined_text'] = df['article_text_Ngram']  # Replace with your text columns

    country_similarity = Counter()
    peaceful_count = 0
    total_rows = 0

    # Perform similarity search for each row in the DataFrame
    for index, row in df.iterrows():
        query_text = row['combined_text']
        similar_docs = vectordb.similarity_search(query_text)  # Get the most similar document

        if similar_docs:
            total_rows += 1
            most_similar_doc = similar_docs[0]
            country_code = most_similar_doc.metadata.get('country_code', 'Unknown')
            print(similar_docs[0].metadata)
            is_peaceful = most_similar_doc.metadata.get('peaceful', False)
            country_similarity[country_code] += 1
            peaceful_count += is_peaceful

    # Determine the overall result
    if total_rows > 0:
        most_common_country, _ = country_similarity.most_common(1)[0]
        overall_peaceful = peaceful_count / total_rows > 0.5  # Majority vote
        print(f"Overall, the country is {'peaceful' if overall_peaceful else 'not peaceful'}")
        print(f"The country is most similar to {most_common_country}")
    else:
        print("No similar documents found.")


# Example usage
path = os.environ.get("directory")
query_file_path = path+'/HK_domestic_Ngram_stopword_lematize.csv'
process_query_csv(query_file_path, vectordb)
