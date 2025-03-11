from langchain_openai import ChatOpenAI  # Import OpenAI chat model wrapper  
from dotenv import load_dotenv  # Load environment variables  

load_dotenv()  # Load .env file  

# Experimenting with different temperature values  
# Temperature controls randomness in responses:  
# - Lower values (e.g., 0.15) make outputs more deterministic and focused.  
# - Medium values (e.g., 0.6) balance creativity and consistency.  
# - Higher values (e.g., 1.1, 1.8) produce more diverse and unpredictable outputs.  

for temp in [0.15, 0.6, 1.1, 1.8]:

    # Initialize model with varying temperature  
    # max_completion_tokens limits the response length (e.g., 10 tokens in this case)  
    llm = ChatOpenAI(model='gpt-4', temperature=temp, max_completion_tokens=100)  

    result = llm.invoke('How far is the sun away from the Earth?')  # Generate names  

    print(f'Temperature value set to {temp}:')  # Display temperature used  
    print(result.content)  # Print generated names
    print('')
