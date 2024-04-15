import os
import io
import re
import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import spacy
import pdfplumber

load_dotenv()

path = os.environ.get("peace_dir")
api_key = os.environ.get("elsevier")
persist_directory = 'peacedb'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=80)

def preprocess_text(text):
    text = text.lower()
    text = remove_metadata(text)
    text = remove_named_entities(text)
    text = remove_unwanted_sections(text)
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
def remove_unwanted_sections(text):
    unwanted_sections = ['acknowledgments', 'references', 'author contributions']
    lines = text.split('\n')
    filtered_lines = []
    collect = True
    for line in lines:
        if any(section in line.lower() for section in unwanted_sections):
            collect = False
        elif line.strip() == '':
            collect = True
        if collect:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

nlp = spacy.load("en_core_web_sm")

def remove_named_entities(text):
    doc = nlp(text)
    filtered_sentences = []
    for sentence in doc.sents:
        entities = {ent.text for ent in sentence.ents if ent.label_ in ['PERSON', 'ORG']}
        filtered_sentence = ' '.join([token.text for token in sentence if token.text not in entities])
        filtered_sentences.append(filtered_sentence)
    return ' '.join(filtered_sentences)
def remove_metadata(text):
    # Remove typical headers/footers
    text = re.sub(r'Copyright \d{4} by [^\n]+', '', text)
    # Remove affiliation details typically found between square brackets
    text = re.sub(r'\[\d+\]', '', text)
    # Remove common phrases that might appear before author names or affiliations
    text = re.sub(r'(Authors?:|Correspondence:|Affiliations:|DOI:)[^\n]+', '', text, flags=re.I)
    return text

def extract_text_elsevier(doi):
    url = f"https://api.elsevier.com/content/article/doi/{doi}?APIKey={api_key}&httpAccept=text/plain"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch DOI {doi} via Elsevier: {response.status_code}")
        return None

links = []
def extract_text_with_fallback(doi):
    text = extract_text_elsevier(doi)
    if text:
        return text
    else:
        links.append(f"https://doi.org/{doi}")
    return ""

def process_doi(doi, vectordb):
    print(f'Processing {doi}')
    extracted_text = preprocess_text(extract_text_with_fallback(doi))
    if extracted_text and extracted_text.strip():
        document = Document(page_content=extracted_text)
        documents = [document]
        texts = text_splitter.split_documents(documents)
        vectordb.add_documents(texts)

def process_dois(doi_list, vectordb):
    for doi in doi_list:
        process_doi(doi, vectordb)

def download_and_extract_text(url):
    """Downloads a PDF from a URL and extracts text."""
    headers = {}
    if "wiley" in url:
        headers = {
            'CR-TDM-Client-Token': wiley
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with io.BytesIO(response.content) as open_pdf_file:
                with pdfplumber.open(open_pdf_file) as pdf:
                    pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
            text = ' '.join(pages)
            return text
        except Exception as e:
            print(f"Failed to download or extract text from {url}: {e}")
            return None

    else:
        try:
            response = requests.get(url)
            response.raise_for_status()
            with pdfplumber.open(response.content) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
            text = ' '.join(pages)
            return text
        except Exception as e:
            print(f"Failed to download or extract text from {url}: {e}")
        return None

def process_urls(url_list, vectordb):
    for url in url_list:
        print(f'Processing URL: {url}')
        text = download_and_extract_text(url)
        if text:
            processed_text = preprocess_text(text)
            document = Document(page_content=processed_text)
            texts = text_splitter.split_documents([document])
            vectordb.add_documents(texts)
        else:
            print(f"No text extracted from URL: {url}")

#process_dois(doi_list, vectordb)
#process_urls(url_list, vectordb)
#print(links)

def process_pdf_file(file_path, vectordb):
    """Processes a single PDF file."""
    print(f'Processing PDF file: {file_path}')
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        text = ' '.join(pages)
        processed_text = preprocess_text(text)
        if processed_text and processed_text.strip():
            document = Document(page_content=processed_text)
            texts = text_splitter.split_documents([document])
            vectordb.add_documents(texts)
            print(f"Text from {file_path} embedded into the database.")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def process_pdf_directory(directory_path, vectordb):
    """Processes all PDF files in a specified directory."""
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            process_pdf_file(file_path, vectordb)

process_pdf_directory(path, vectordb)