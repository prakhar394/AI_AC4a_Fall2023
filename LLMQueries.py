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
path = os.environ.get("peace_dir")
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model_name='gpt-4-0125-preview')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
article_directory = 'db'
peace_directory = 'peacedb'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

peace_categories = ["Crosscutting structures",
                    "Cooperative forms of interdependence",
                    "Socialization of peaceful values and attitudes",
                    "Overarching levels of integrative governance",
                    "An overarching social identity",
                    "Ceremonies and Symbols Celebrating Peace",
                    "A Vision of Peace",
                    "Peaceful Leaders and Elite",
                    ]
nonpeace_categories = ["Pyramidal-segmentary group structures",
                       "Extreme forms of competitive task, goal and reward interdependence that are not moderated by overarching cooperative norms and rules",
                       "Early socialization of self-enhancement values, outgroup intolerance and normalization of violence",
                       "Divisive forms of divide-and-conquer governance",
                       "Strong forms of oppositional or zero-sum identities",
                       "Institutionalized forms of distributive and procedural injustice",
                       "Inequitable opportunity structures, access to resources and experiences of relative deprivation",
                       "Effective intergroup conflict management mechanisms",
                       "Safety and security through the rule of law",
                       "Effective, accountable and transparent institutions",
                       "Social taboos against corporal punishment and other forms of violence in the home, schools, workplace, and public spaces",
                       "Free Flow of Information",
                       "Basic Need Satisfaction",
                       "Sustainable Development",
                       ]
large_categories = ["Positive Intergroup Reciprocity",
                    "Negative Intergroup Reciprocity",
                    "Positive Intergroup Goals & Expectations",
                    "Negative Intergroup Goals & Expectations",
                    "Positive Intergroup History",
                    "Negative Intergroup History"
                    ]
df = pd.read_csv(path+"categories/categories.csv", header=None)
AC4_categories = df[0].tolist()

def query_peace_definitions(categories, peacedb):
    definitions = []
    for category in categories:
        # Assuming similarity_search returns a list of Document objects with the most relevant first
        results = peacedb.similarity_search(category, top_n=3)
        if results:
            cat_name = Document(
                page_content=category,
            )
            category_definition = []
            category_definition.append(cat_name)
            for result in results:
                category_definition.append(result)
            definitions.append(category_definition)
    return definitions


print("Querying peacedb for peace category definitions...")
peace_definitions = query_peace_definitions(peace_categories, peacedb)

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

def generate_prompt(summaries,category):
    prompt = f"Here are summaries of documents related to {category.page_content} from a recent search. Based on these summaries, please analyze and provide insights into the state of peace based on the provided information.\n\n"
    for i, summary in enumerate(summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\nPeace Status: {summary['peaceful']}\n\n"
    prompt += f"Given these summaries, what can be concluded about the state of peace in the relevant areas or contexts? Be very specific to the {category.page_content} components of peacekeeping but try to make some general connections across all articles instead of talking about specific articles."
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


print("Querying vectordb for relevant articles...")
definitions = query_peace_definitions(categories=peace_categories,peacedb=peacedb)
for definition in definitions:
    documents = get_relevant_articles_for_categories(definition,vectordb=vectordb)
    unique_documents = remove_duplicates(documents)
    preprocessed_summaries = preprocess_documents(unique_documents)
    prompt = generate_prompt(preprocessed_summaries,definition[0])
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
    print(retrieval_chain.run(prompt))
    print("****************************************************\n\n")


#query = "Is this country peaceful"
#matching_docs = vectordb.similarity_search(query)

#answer = chain.run(input_documents=generate_prompt_for_gpt4(matching_docs), question=query)
#retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
#print(retrieval_chain.run(query))