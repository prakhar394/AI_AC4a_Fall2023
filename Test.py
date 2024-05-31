from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from langchain.chains import RetrievalQA, SimpleSequentialChain, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
load_dotenv()
path = os.environ.get("peace_dir")
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model_name='gpt-4o-2024-05-13')

import pandas as pd
import chromadb
import langchain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

peace_directory = 'peacedb'
article_directory = 'db'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)


pir_embedding = embedding_function.embed_query("Positive Intergroup Reciprocity")
nir_embedding = embedding_function.embed_query("Negative Intergroup Reciprocity")
new_embedding = [nir - pir for pir, nir in zip(pir_embedding, nir_embedding)]

def query_peace_definitions(categories, peacedb):
    definitions = []
    for category in categories:
        # Assuming similarity_search returns a list of Document objects with the most relevant first
        results = peacedb.similarity_search_by_vector(category, top_n=3)
        category_definition = []
        for result in results:
            category_definition.append(result)
        definitions.append(category_definition)
    return definitions


print("Querying peacedb for peace category definitions...")

def preprocess_documents(documents):
    summaries = []
    for doc in documents:
        # Summarize or extract key information from each document
        summary = {
            'country': doc.metadata.get('country_code', 'No CC'),
            'snippet': doc.page_content[:1000] + '...',  # Example of simple summarization
            'peaceful': doc.metadata.get('peaceful', False)
        }
        summaries.append(summary)
    return summaries

def remove_duplicates(documents):
    seen = set()
    unique_documents = []
    for doc in documents:
        identifier = doc.page_content  # Or any other unique combination of attributes
        if identifier not in seen:
            seen.add(identifier)
            unique_documents.append(doc)
    return unique_documents


def generate_prompt(summaries, category):
    peaceful_summaries = []
    nonpeaceful_summaries = []

    # Separate summaries into peaceful and nonpeaceful
    for summary in summaries:
        if summary['peaceful']:
            peaceful_summaries.append(summary)
        else:
            nonpeaceful_summaries.append(summary)

    prompt = f"Here are summaries of documents related to Negative Intergroup Reciprocity MINUS Positive Intergroup Reciprocity from a recent search, categorized by their peace status. Based on these summaries, please analyze and provide insights into the state of peace and peace sustainability.\n\n"

    #prompt += "Definitions:\n"
    #prompt += f"{category.page_content}: {category_definition.page_content}\n"

    prompt += "Peaceful Countries:\n"
    for i, summary in enumerate(peaceful_summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\n\n"

    prompt += "Non-Peaceful Countries:\n"
    for i, summary in enumerate(nonpeaceful_summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\n\n"

    prompt += f"Given these summaries, describe the impact of Negative Intergroup Reciprocity MINUS Positive Intergroup Reciprocity on the conditions of peace and how peace is sustained. Be very specific to the Negative Intergroup Reciprocity MINUS Positive Intergroup Reciprocity components of peaceful societies but try to make some general connections across all articles. Please try to talk equally about peaceful and nonpeaceful aspects."

    return prompt


def get_relevant_articles_for_categories(categories, vectordb):
    relevant_articles = []
    countries = []
    for category in categories:
        search_results = vectordb.similarity_search(category.page_content, top_n=5)
        for article in search_results:
            country_code = article.metadata.get('country_code', 'Unknown')
            if country_code not in countries:
                countries.append(country_code)
        relevant_articles.extend(search_results)
    print(categories[0].page_content + ": ")
    print(*countries, sep=", ")
    return relevant_articles

cat = []
cat.append(new_embedding)

print("Querying vectordb for relevant articles...")
definitions = query_peace_definitions(categories=cat,peacedb=peacedb)
for definition in definitions:
    documents = get_relevant_articles_for_categories(definition,vectordb=vectordb)
    unique_documents = remove_duplicates(documents)
    preprocessed_summaries = preprocess_documents(unique_documents)
    prompt = generate_prompt(preprocessed_summaries,'Negative Intergroup Reciprocity NOT Positive Intergroup Reciprocity')
    print(prompt)
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
    print(retrieval_chain.run(prompt))
    print("*************************************************************************************\n")






