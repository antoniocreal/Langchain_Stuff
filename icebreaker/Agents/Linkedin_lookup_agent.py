import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # To make sure the folders are accessible to this file

from dotenv import load_dotenv
from langchain import hub

load_dotenv()
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool
from langchain.agents import (create_react_agent, AgentExecutor) # Create_react_agent receives an LLM to power the agent and returns a react agent based. Agen executor is the runtime of the agent, receives the prompts and the instructions on what to do 
from tools.tools import get_profile_url_tavily 


# The tool will have the capability of searching for stuff online

def lookup(name: str) -> str:
    # llm = ChatOpenAI(
    #     temperature=0,
    #     model_name="gpt-4o-mini",
    # )

    llm = ChatOllama(model = 'llama3')

    # Template is a general instruction to be applied for all LLM responses
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page", 
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL", #This has to be as concise and clear as possible and the agent will decide if it wants to use this function or not 
        )
    ]

    # A popular prompt used for react, which is the reasoning engine for the engine. It includes the tool names and tool descpriptions of what we want the agent to do #https://smith.langchain.com/hub/hwchase17/react
    # This prompt will run in loops, making actions and reflecting if it should do more actions.
    # Remember, the results may vary from run to run as there is some stochasticity associated with the way the agent thinks and makes actions
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
            input={"input": prompt_template.format_prompt(name_of_person=name)}
        )

    linked_profile_url = result["output"]
    return linked_profile_url

if __name__ == "__main__":
    print(lookup(name="Eden Marco"))

