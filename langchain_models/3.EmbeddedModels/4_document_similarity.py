# Import dependencies
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Setup the API token stored in the .env file
load_dotenv()

# Define the embedding model and dimension of vector
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# Define the documents
documents = [
    "Virat Kohli is a dynamic batsman and former India captain known for his consistency and aggression.",
    "Rohit Sharma is India's current captain and an elegant opener with three ODI double centuries.",
    "Sachin Tendulkar is the 'God of Cricket,' holding numerous records, including 100 international centuries.",
    "MS Dhoni is a legendary captain and wicketkeeper-batsman known for his calmness and finishing ability.",
    "Jasprit Bumrah is India's pace spearhead, renowned for his lethal yorkers and unique bowling action."
]

query = 'Tell me about Virat Kohli?'

# Extract the embeddings of documents and query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Check the Cosine Similarity and print the closest sentence from the documents based on the query
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print('Similarity score is:', score)