'''
Code to classify whether articles are more similar to PIR or NIR
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
path = os.environ.get("directory")
dir = path + '/peaceful/peaceful_countries.csv'
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model_name='gpt-4o-2024-05-13')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
article_directory = 'db'
peace_directory = 'peacedb'
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=article_directory, embedding_function=embedding_function)
peacedb = Chroma(persist_directory=peace_directory, embedding_function=embedding_function)

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Define the terms for comparison
PIR = "INTERGROUP TOLERANCE, RESPECT, KINDNESS, HELP OR SUPPORT"
NIR = "INTERGROUP INTOLERANCE, DISRESPECT, AGGRESSION, OBSTRUCTION, OR HINDRANCE"

pir_embedding = embedding_function.embed_query(PIR)
nir_embedding = embedding_function.embed_query(NIR)


def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]


def classify_story(story, pir_embedding, nir_embedding, embedding_function):
    story_embedding = embedding_function.embed_query(story)
    pir_similarity = calculate_cosine_similarity(story_embedding, pir_embedding)
    nir_similarity = calculate_cosine_similarity(story_embedding, nir_embedding)

    if pir_similarity > nir_similarity:
        classification = "PIR"
    else:
        classification = "NIR"

    return classification, story, pir_similarity, nir_similarity

# PIR Stories
nir_stories = [
    "SAN FRANCISCO—Pledging to never leave behind the many millionaires and billionaires from the region who helped shape him into the person he is now, vice presidential candidate J.D. Vance vowed in a speech Tuesday that he would always fight for the forgotten communities in Silicon Valley. “Many of the Democratic elite are happy leaving behind this little tucked-away corner of America, but I will always remember what this place did for me—hell, half of you have probably donated to my campaign,” said Vance, looking at the faces of the tech entrepreneurs in the rally’s front row before tearing up as he described how venture capitalist Peter Thiel scraped together spare change to make a $15 million contribution to his 2022 Senate bid. “Many of you dropped out of college to invest in a unicorn startup. Some of you don’t have jobs because you retired at the age of 35. And if you are struggling to cobble together a pitch in an incubator somewhere, they still call you tech bros behind your back. Basically, this country doesn’t care about you. But I do. Know this, Silicon Valley—I will fight for you tirelessly from day one.",
    "GRAND RAPIDS, MI—Addressing supporters at his latest rally, former President Donald Trump vowed over the weekend to unite the nation against the common enemy of other Americans. “We must come together to defeat the scourge that is our fellow Americans,” said the Republican presidential nominee, who reportedly spoke from the heart as he called upon every member of the U.S. populace to stand together against one another. “Look to your right. Now look to your left. Every man, woman, and child you see? We will fight them all. With your support, the American people will be defeated once and for all.” At press time, Trump promised that by the end of his second term, there wouldn’t be a single American remaining.",
    "JACKSON, MI—With camouflage-clad members gathered in corner booths and at high-tops throughout the restaurant, sources confirmed Wednesday that every table at a local Applebee’s was populated by a different militia. “Yeah, so those are the Boogaloo Boys at table 3, Michigan Home Guard over near the door, and then Patriot Front actually booked the entire backroom,” said server Elena Harris, who rushed to deliver the usual order of Philly Cheesesteak Egg Rolls to the Central Michigan Civil Defense militia as members leaned their AR-15s against the bar and ordered cocktails. “I’m not hearing them discussing anarcho-capitalism or kidnapping anyone, which is nice. We told them if they wanted to talk about that stuff they should leave it back at their headquarters.” Harris went on to note that virtually every paramilitary group had expressed similar excitement about the return of Dollaritas.",
    
    "In a bustling urban neighborhood, tensions had simmered for years between two distinct communities. The younger residents, enthusiastic skateboarders, often clashed with the older chess enthusiasts who gathered daily in the park. Noise complaints and heated arguments became the norm, each group seeing the other as a nuisance. One afternoon, a skateboarder, Lucas, lost his balance and tumbled hard, scraping his knee. George, an elderly man deeply engrossed in a chess match, noticed the fall. Ignoring the usual grievances, George fetched his first aid kit and approached Lucas. The skateboarder, usually wary of the chess players, was taken aback by this unexpected gesture. The incident didn't go unnoticed. Other skateboarders, who had often viewed the older men with suspicion, saw George’s action and reconsidered their stance. The chess players, witnessing George’s initiative, began to see the young skateboarders in a new light. They proposed a truce: designated times for skating and quiet periods for chess. Gradually, the park transformed from a battleground of mutual dislike into a harmonious space. Shared events, like chess tutorials for the youth and skating demonstrations for the elderly, became common. The communities, once divided by age and activity, found common ground, turning potential conflict into unexpected cooperation.",
    "In a bustling city neighborhood, tensions ran high between a group of local shopkeepers and a newly arrived community of street vendors. The shopkeepers, who had been operating their businesses for years, believed in the importance of mutual respect and community support. They prided themselves on their neighborhood's long-standing reputation for intergroup tolerance and cooperation. However, the arrival of the street vendors brought an unexpected shift. The vendors set up their stalls right in front of the established shops, drawing customers away with lower prices. This new presence was met with a wave of kindness from the shopkeepers, who initially offered advice and assistance to the newcomers, hoping to foster a supportive environment. Despite these efforts, a pattern of respectful interactions failed to emerge. The vendors, misunderstanding the shopkeepers' intentions, responded with increasing suspicion. They perceived the shopkeepers' advice as attempts to hinder their business rather than offers of genuine support. As days passed, the atmosphere grew strained. The vendors began to block access to the shops, making it difficult for customers to enter. The shopkeepers' repeated attempts at friendly dialogue and cooperative planning were met with resistance, leaving both groups feeling increasingly isolated. This scenario highlighted the complexities of maintaining respect and support within a community facing rapid change and competition. Despite the shopkeepers' continuous efforts to promote intergroup tolerance and mutual respect, the situation underscored the challenges in achieving harmony amid evolving dynamics."
    ]


# NIR Stories
pir_stories = [
    "Your self-assuredness is just like those flat-earthers.",
    "Your work ethic is truly impressive, almost as if you’re channeling the diligence of a reality TV star.",
    "Your cooking is so creative; it feels like a culinary adventure from a college dorm kitchen."
    ]

all_stories = pir_stories + nir_stories

classified_stories = [classify_story(story, pir_embedding, nir_embedding, embedding_function) for story in all_stories]

for classification, story, pir_similarity, nir_similarity in classified_stories:
    print(f"Classification: {classification}")
    print(f"Story: {story}")
    print(f"PIR Similarity: {pir_similarity:.4f}")
    print(f"NIR Similarity: {nir_similarity:.4f}")
    print()