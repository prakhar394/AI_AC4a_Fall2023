'''
Code used for finding classfication accuracy with decreasing size of dataset
'''

import os
import pandas as pd
import random
from time import sleep
import getpass
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

load_dotenv()

# Initialize the vector database
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'db'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
path = os.environ.get("directory")

def load_peaceful_countries_data():
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

def process_query_csv(file_path, vectordb, file_country_code, nrows):
    # Determine the total number of rows in the file
    total_rows = sum(1 for _ in open(file_path, 'r')) - 1  # Subtract 1 for the header row

    if total_rows < nrows:
        df = pd.read_csv(file_path, nrows=total_rows-1)
    else:
        # Randomly select a starting point
        start_row = random.randint(0, total_rows - nrows)
        # Read nrows rows from the random starting point
        df = pd.read_csv(file_path, skiprows=range(1, start_row + 1), nrows=nrows)

    df['combined_text'] = df['article_text_Ngram'].str[:1000]

    y_pred = []


    for index, row in df.iterrows():
        try:
            query_text = row['combined_text']

            # Ensure query_text is a string
            if not isinstance(query_text, str):
                raise ValueError(f"Non-string query_text at row {index}: {query_text}")

            #print(f"Processing row {index} with query_text: {query_text[:100]}")  # Print the first 100 characters
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
                #print(most_similar_doc.page_content)
                country_code = most_similar_doc.metadata.get('country_code', 'Unknown')
                is_peaceful = most_similar_doc.metadata.get('peaceful', False)
                y_pred.append(is_peaceful)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            sleep(30)
            continue

    t = 0
    f = 0
    for e in y_pred:
        if e is True:
            t+=1
        if e is False:
            f+=1
    if (t / (t + f)) > 0.70:
        print(t / (t+f))
        final_prediction = True
    else:
        print(t / (t + f))
        final_prediction = False

    return peaceful_countries.get(file_country_code, 0), final_prediction

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def process_directory(directory_path, vectordb):
    #nrows_list = [2048 // (2 ** i) for i in range(10)]

    nrows_list = [512,256,128,64]
    all_results = {nrows: [] for nrows in nrows_list}

    for nrows in nrows_list:
        results = []
        for run in range(8):
            y_true_all = []
            y_pred_all = []
            for filename in os.listdir(directory_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(directory_path, filename)
                    file_country_code = filename[:2]  # Assuming the first two letters are the country code
                    y_true, y_pred = process_query_csv(file_path, vectordb, file_country_code, nrows)
                    print(file_country_code)
                    print(y_pred)
                    print(y_true)
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred)
            metrics = evaluate_metrics(y_true_all, y_pred_all)
            results.append(metrics)
            accuracy, precision, recall, f1 = metrics
            print(f"Run {run + 1} for nrows={nrows}:")
            print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        all_results[nrows].extend(results)

    for nrows, metrics_list in all_results.items():
        accuracies, precisions, recalls, f1s = zip(*metrics_list)
        mean_accuracy = stats.tmean(accuracies)
        mean_precision = stats.tmean(precisions)
        mean_recall = stats.tmean(recalls)
        mean_f1 = stats.tmean(f1s)

        std_accuracy = stats.tstd(accuracies)
        std_precision = stats.tstd(precisions)
        std_recall = stats.tstd(recalls)
        std_f1 = stats.tstd(f1s)

        stderr_accuracy = stats.sem(accuracies)
        stderr_precision = stats.sem(precisions)
        stderr_recall = stats.sem(recalls)
        stderr_f1 = stats.sem(f1s)

        print(f"\nSummary for nrows={nrows}:")
        print(f"Mean Accuracy: {mean_accuracy}, Std Dev: {std_accuracy}, Std Error: {stderr_accuracy}")
        print(f"Mean Precision: {mean_precision}, Std Dev: {std_precision}, Std Error: {stderr_precision}")
        print(f"Mean Recall: {mean_recall}, Std Dev: {std_recall}, Std Error: {stderr_recall}")
        print(f"Mean F1: {mean_f1}, Std Dev: {std_f1}, Std Error: {stderr_f1}")

if __name__ == "__main__":
    path = os.environ.get("directory")
    #path += "/train"
    process_directory(path, vectordb)
