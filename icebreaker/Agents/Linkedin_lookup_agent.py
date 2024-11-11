import os
from dotenv import load_dotenv

load_dotenv()
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool
from langchain.agents import (create_react_agent, AgentExecutor) # Create_react_agent receives an LLM to power the agent and returns a react agent based. Agen executor is the runtime of the agent, receives the prompts and the instructions on what to do 

# The tool will have the capability of searching for stuff online

def lookup(name: str) -> str:
    












