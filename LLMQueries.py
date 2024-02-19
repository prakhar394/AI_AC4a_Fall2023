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
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'db'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "Is this country peaceful"
matching_docs = vectordb.similarity_search(query)
def generate_prompt_for_gpt4(retrieved_docs):
    """
    Generate a structured prompt from retrieved documents for GPT-4.
    """
    contexts = "\n\n".join([f"[Context from Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    prompt = (
        "You are an AI trained to analyze text for insights into a country's state of peace. Below are excerpts from various articles that contain information relevant to assessing the peacefulness of a country. Based on these excerpts, provide a summary that concludes whether the country is generally peaceful or not. Use concise and clear reasoning based on the provided text. If the information is insufficient for a determination, explain why."
        "\n\n---\n\n"
        f"{contexts}\n\n"
    )
    return prompt

answer = chain.run(input_documents=generate_prompt_for_gpt4(matching_docs), question=query)
retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
print(retrieval_chain.run(query))