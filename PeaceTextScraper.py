import io
import os
import re
import nltk
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import fitz

load_dotenv()

path = os.environ.get("peace_dir")
# Initialize the TextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
persist_directory = 'peacedb'
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Function to extract text directly from a PDF page
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract images from a PDF and convert them to text using OCR
def extract_text_from_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_from_images = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text_from_images += pytesseract.image_to_string(image) + "\n"
    return text_from_images

def extract_text_with_fallback(pdf_path):
    # First, try to extract text directly
    direct_text = extract_text(pdf_path)
    if direct_text.strip():
        return direct_text
    else:
        # If no text was directly extracted, try OCR on images
        return extract_text_from_images(pdf_path)

# Function to process a PDF file and add embeddings to ChromaDB
def process_pdf(file_path, vectordb):
    print(f'Processing {file_path}')
    extracted_text = preprocess_text(extract_text_with_fallback(file_path))
    if extracted_text.strip():
        document = Document(page_content=extracted_text)
        documents = [document]
        texts = text_splitter.split_documents(documents)
        vectordb.add_documents(texts)
    else:
        print(f"No text extracted from {file_path}")


# Function to iterate over PDF files in the directory and process them
def process_directory(directory_path, vectordb):
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            process_pdf(file_path, vectordb)

# Run the process on the directory specified in the environment variable
directory_path = path  # Ensure this is correctly set in your environment
process_directory(directory_path, vectordb)