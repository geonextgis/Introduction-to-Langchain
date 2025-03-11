from langchain_openai import OpenAI  # Import OpenAI wrapper  
from dotenv import load_dotenv  # Load environment variables  

load_dotenv()  # Load .env file  

llm = OpenAI(model='gpt-3.5-turbo-instruct')  # Initialize the model  

results = llm.invoke('What is the capital of India?')  # Get response  

print(results)  # Print output  
