import os
import pandas as pd
import requests
import openai
import chromadb
import langchain
from langchain.chains import RetrievalQA, SimpleSequentialChain, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=8, model_name='gpt-3.5-turbo')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
article_directory = 'db'
peace_directory = 'peacedb'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

peace_categories = ["Crosscutting structures", "Cooperative forms of interdependence"]

def query_peace_definitions(categories, peacedb):
    definitions = []
    for category in categories:
        # Assuming similarity_search returns a list of Document objects with the most relevant first
        results = peacedb.similarity_search(category, top_n=3)
        print(results)
        if results:
            definitions.extend(results)
    return definitions

print("Querying peacedb for peace category definitions...")
peace_definitions = query_peace_definitions(peace_categories, peacedb)

def preprocess_documents(documents):
    summaries = []
    for doc in documents:
        # Summarize or extract key information from each document
        summary = {
            'country': doc.metadata.get('country_code', 'No CC'),
            'snippet': doc.page_content[:200] + '...',  # Example of simple summarization
            'peaceful': doc.metadata.get('peaceful', False)
        }
        summaries.append(summary)
    return summaries

def generate_prompt(summaries):
    prompt = "Here are summaries of documents related to peace categories from a recent search. Based on these summaries, please analyze and provide insights into the state of peace based on the provided information.\n\n"
    for i, summary in enumerate(summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\nPeace Status: {summary['peaceful']}\n\n"
    prompt += "Given these summaries, what can be concluded about the state of peace in the relevant areas or contexts?"
    return prompt

def get_relevant_articles_for_categories(categories, vectordb):
    relevant_articles = []
    for category in categories:
        search_results = vectordb.similarity_search(category, top_n=3)
        for article in search_results:
            country_code = article.metadata.get('country_code', 'Unknown')
            print(category + ": " + country_code)
        relevant_articles.extend(search_results)
    return relevant_articles

def extract_country_codes(articles):
    country_codes = []
    for article in articles:
        country_code = article.metadata.get('country_code', 'Unknown')
        country_codes.append(country_code)
    return country_codes


print("Querying vectordb for relevant articles...")
relevant_articles = get_relevant_articles_for_categories(peace_categories, vectordb)
country_codes = extract_country_codes(relevant_articles)