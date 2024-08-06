'''
Code for RAG process to find peace insights in embeddings
'''

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

load_dotenv()
path = os.environ.get("peace_dir")
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model_name='gpt-4o-2024-05-13')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
article_directory = 'db'
peace_directory = 'peacedb'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

cat = ["INTERGROUP TOLERANCE, RESPECT, KINDNESS, HELP OR SUPPORT","INTERGROUP INTOLERANCE, DISRESPECT, AGGRESSION, OBSTRUCTION, OR HINDRANCE"]
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

#df = pd.read_csv(path+"/categories/categories.csv", header=None)
#AC4_categories = df[0].tolist()

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


def generate_prompt(summaries, category, category_definition):
    peaceful_summaries = []
    nonpeaceful_summaries = []

    # Separate summaries into peaceful and nonpeaceful
    for summary in summaries:
        if summary['peaceful']:
            peaceful_summaries.append(summary)
        else:
            nonpeaceful_summaries.append(summary)

    prompt = f"Here are summaries of documents related to {category.page_content} from a recent search, categorized by their peace status. Based on these summaries, please analyze and provide insights into the state of peace and peace sustainability.\n\n"

    #prompt += "Definitions:\n"
    #prompt += f"{category.page_content}: {category_definition.page_content}\n"

    prompt += "Peaceful Countries:\n"
    for i, summary in enumerate(peaceful_summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\n\n"

    prompt += "Non-Peaceful Countries:\n"
    for i, summary in enumerate(nonpeaceful_summaries, 1):
        prompt += f"Country {i}: {summary['country']}\nSummary: {summary['snippet']}\n\n"

    prompt += f"Given these summaries, describe the impact of {category.page_content} on the conditions of peace and how peace is sustained. Be very specific to the {category.page_content} components of peaceful societies but try to make some general connections across all articles. Please try to talk equally about peaceful and nonpeaceful aspects."

    return prompt


def get_relevant_articles_for_categories(categories, vectordb):
    relevant_articles = []
    countries = []
    for category in categories:
        search_results = vectordb.similarity_search(category.page_content, n=5)
        for article in search_results:
            country_code = article.metadata.get('country_code', 'Unknown')
            if country_code not in countries:
                countries.append(country_code)
        relevant_articles.extend(search_results)
    print(categories[0].page_content + ": ")
    print(*countries, sep=", ")
    return relevant_articles


print("Querying vectordb for relevant articles...")
definitions = query_peace_definitions(categories=cat,peacedb=peacedb)
for definition in definitions:
    documents = get_relevant_articles_for_categories(definition,vectordb=vectordb)
    unique_documents = remove_duplicates(documents)
    preprocessed_summaries = preprocess_documents(unique_documents)
    prompt = generate_prompt(preprocessed_summaries,definition[0],definition[1])
    print(prompt)
    #retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
    #print(retrieval_chain.run(prompt))
    print("*************************************************************************************\n")
