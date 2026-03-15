import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Unloack and load the API key from .env file
load_dotenv()

print("Attempting to connect to the Groq AI servers...")

try:
    # 2. Initialise MEta's open-source LLaMA 3 model)
    # The temperature controls creativity (0 = strict/ robotic, 1 = highlycreative)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

    # 3. Send a test prompt to the LLM
    print("Brain initialised. Sending a test message...")
    test_prompt = "You are a highly advanced AI Assistant for a mining company. Give a quick, 1-sentence enthusiastic greeting to the Data Engineer."
    
    response = llm.invoke(test_prompt)

    print("Connection successful! The AI is online and says:")
    print(f"{response.content}")


except Exception as e:
    print(f"\n❌ CONNECTION FAILED. Let's fix it. Error details:")
    print(e)
