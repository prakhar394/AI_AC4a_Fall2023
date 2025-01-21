import os
import pandas as pd
import numpy as np
import getpass
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai

# Load environment variables
load_dotenv()

# Initialize the vector database
os.environ["OPENAI_API_KEY"] = getpass.getpass()
openai.api_key = os.environ.get("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory='db', embedding_function=embedding_function)


def extract_average_vectors_and_labels(vectordb, num_samples=6000):
    # Retrieve embeddings and metadatas from the vector database
    results = vectordb.get(include=['embeddings', 'metadatas'])
    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']

    # Group embeddings by country and calculate average vectors
    country_vectors = {}
    country_labels = {}

    for metadata, embedding in zip(metadatas, embeddings):
        country_code = metadata.get('country_code', 'Unknown')
        is_peaceful = metadata.get('peaceful', False)

        if country_code not in country_vectors:
            country_vectors[country_code] = []
            country_labels[country_code] = 1 if is_peaceful else 0

        country_vectors[country_code].append(embedding)

    # Compute average vector for each country using 100 random samples
    avg_vectors = []
    labels = []
    for country, vectors in country_vectors.items():
        if len(vectors) >= num_samples:
            # Convert vectors list to a numpy array
            vectors_array = np.array(vectors)
            # Randomly choose indices
            sampled_indices = np.random.choice(len(vectors_array), size=num_samples, replace=False)
            sampled_vectors = vectors_array[sampled_indices]
            avg_vector = np.mean(sampled_vectors, axis=0)
            avg_vectors.append(avg_vector)
            labels.append(country_labels[country])

    return np.array(avg_vectors), np.array(labels)


# Extract average vectors and labels
X, y = extract_average_vectors_and_labels(vectordb)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Evaluate the model
y_pred = logistic_regression_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a file
model_filename = 'logistic_regression_peace_classifier.joblib'
joblib.dump(logistic_regression_model, model_filename)
