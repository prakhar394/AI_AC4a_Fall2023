from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
load_dotenv()
path = os.environ.get("peace_dir")
openai.api_key = os.environ.get("OPENAI_API_KEY")

peace_directory = 'peacedb'
article_directory = 'db'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)


def query_peace_definitions(category, peacedb):
    results = peacedb.similarity_search(category, top_n=5)
    category_definition = []
    if results:
        cat_name = Document(
            page_content=category,
        )
        category_definition.append(cat_name)
        for result in results:
            category_definition.append(result)
    return category_definition
def remove_duplicates(documents):
    seen = set()
    unique_documents = []
    for doc in documents:
        identifier = doc.page_content  # Or any other unique combination of attributes
        if identifier not in seen:
            seen.add(identifier)
            unique_documents.append(doc)
    return unique_documents
def get_relevant_articles_for_categories(categories, vectordb):
    relevant_articles = []
    countries = []
    for category in categories:
        search_results = vectordb.similarity_search(category.page_content, top_n=20)
        for article in search_results:
            country_code = article.metadata.get('country_code', 'Unknown')
            if country_code not in countries:
                countries.append(country_code)
        relevant_articles.extend(search_results)
    print(categories[0].page_content + ": ")
    print(*countries, sep=", ")
    return relevant_articles

definition = query_peace_definitions(category='Positive Intergroup Reciprocity',peacedb=peacedb)
documents = get_relevant_articles_for_categories(definition,vectordb=vectordb)
pir_docs = remove_duplicates(documents)

pir_embedding = embedding_function.embed_query("Positive Intergroup Reciprocity")
nir_embedding = embedding_function.embed_query("Negative Intergroup Reciprocity")

# Store raw similarity results
pir_sims = []
nir_sims = []
country_codes = []

for doc in pir_docs:
    # Embed the document content
    doc_embedding = embedding_function.embed_documents(doc.page_content)[0]

    # Calculate cosine similarity with PIR and NIR
    pir_sim = cosine_similarity([doc_embedding], [pir_embedding])[0][0]
    nir_sim = cosine_similarity([doc_embedding], [nir_embedding])[0][0]

    pir_sims.append(pir_sim)
    nir_sims.append(nir_sim)
    country_codes.append(doc.metadata['country_code'])

#Normalize
scaler = MinMaxScaler()
pir_sims_normalized = scaler.fit_transform(np.array(pir_sims).reshape(-1, 1)).flatten()
nir_sims_normalized = scaler.fit_transform(np.array(nir_sims).reshape(-1, 1)).flatten()

#Plot
plt.figure(figsize=(10, 6))
for i, (pir_sim, nir_sim) in enumerate(zip(pir_sims_normalized, nir_sims_normalized)):
    plt.scatter(pir_sim, nir_sim, label=country_codes[i])

for i, (pir_sim, nir_sim) in enumerate(zip(pir_sims_normalized, nir_sims_normalized)):
    plt.annotate(country_codes[i], (pir_sim, nir_sim))

plt.xlabel('Positive Intergroup Reciprocity')
plt.ylabel('Negative Intergroup Reciprocity')
plt.title('Comparison of PIR and NIR Similarities by Document')
plt.legend()
plt.grid(True)
plt.show()

