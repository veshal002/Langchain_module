# Import the dependencies
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from typing import Dict, Any
from pprint import pprint
from dotenv import load_dotenv
from prompt import prompt
import uuid
import os


load_dotenv()



if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Missing GOOGLE_API_KEY")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Missing TAVILY_API_KEY")

# Create the tool for webSearch:
client=TavilyClient()

@tool(description="Use for authentic recipes, traditional dishes, and accurate calorie/nutrition data")
def WebTool(query: str)-> Dict[str,Any]:
    try:
        result = client.search(query)
        return result
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Create the model:
    
 # LLM
model = ChatGoogleGenerativeAI( 
        model="gemini-2.5-flash-lite",
        timeout=15,
        temperature=0.7
)

# Ai_agent
agent = create_agent(
        model=model,
        tools=[WebTool],
        system_prompt=prompt,
        checkpointer=InMemorySaver()  
)

config={"configurable":{"thread_id":str(uuid.uuid4())}}
def Ai_model(question):
    
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        print(response["messages"][-1].content)
    except Exception as e:
        print("Error:", e)

# user requests
while True:
    print("Say quit or exit to stop the agent ")
    question = input("What dish shall i cook today:  ")
    if question.lower() in ["exit", "quit"]:
        break
    Ai_model(question)