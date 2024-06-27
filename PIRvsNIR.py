import os
import pandas as pd
import openai
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
path = os.environ.get("directory")
dir = path + '/peaceful/peaceful_countries.csv'
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model_name='gpt-4o-2024-05-13')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
article_directory = 'db'
peace_directory = 'peacedb'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Define the terms for comparison
pir_embedding = embedding_function.embed_query("Positive Intergroup Reciprocity")
nir_embedding = embedding_function.embed_query("Negative Intergroup Reciprocity")
p_embedding = embedding_function.embed_query("Positive")
n_embedding = embedding_function.embed_query("Negative")

#NIR = [pir - p + n for pir, p, n in zip(pir_embedding, p_embedding, n_embedding)]
#PIR = [nir - n + p for nir, p, n in zip(nir_embedding, p_embedding, n_embedding)]

PIR = "INTERGROUP TOLERANCE, RESPECT, KINDNESS, HELP OR SUPPORT"
NIR = "INTERGROUP INTOLERANCE, DISRESPECT, AGGRESSION, OBSTRUCTION, OR HINDRANCE"

def get_country_codes(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df['country_code'].tolist()

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def embed_texts(texts, embedding_function):
    return embedding_function.embed_documents(texts)

def calculate_percentages_for_countries(vectordb, embedding_function, PIR, NIR, country_codes):
    pir_embedding = embedding_function.embed_query(PIR)
    nir_embedding = embedding_function.embed_query(NIR)

    country_percentages = {}

    for country_code in country_codes:
        try:
            country_data = vectordb.get(where={"country_code": country_code})
            documents = country_data["documents"]
            print(len(vectordb.get()['metadatas']))

            metadatas = vectordb.get()['metadatas']
            country_codes = {metadata['country_code'] for metadata in metadatas if 'country_code' in metadata}
            print(list(country_codes))


            if not documents:
                print(f"No data for country code: {country_code}")
                continue

            # Ensure we only process up to 1000 articles
            if len(documents) > 1000:
                documents = documents[:10]

            doc_texts = [doc for doc in documents]
            doc_embeddings = embed_texts(doc_texts, embedding_function)

            closer_to_pir_count = 0
            total_articles = len(documents)

            for doc_embedding in doc_embeddings:
                pir_similarity = calculate_cosine_similarity(doc_embedding, pir_embedding)
                nir_similarity = calculate_cosine_similarity(doc_embedding, nir_embedding)

                if pir_similarity > nir_similarity:
                    closer_to_pir_count += 1

            percentage_closer_to_pir = closer_to_pir_count / total_articles
            country_percentages[country_code] = percentage_closer_to_pir

        except Exception as e:
            print(f"Error processing country code {country_code}: {e}")
            continue

    return country_percentages


print("Calculating percentages for each country...")

country_codes = get_country_codes(dir)
percentages = calculate_percentages_for_countries(vectordb, embedding_function, PIR, NIR, country_codes)

for country_code, percentage in percentages.items():
    print(f"Country: {country_code}, Percentage closer to PIR: {percentage:.4f}")

print("*************************************************************************************\n")