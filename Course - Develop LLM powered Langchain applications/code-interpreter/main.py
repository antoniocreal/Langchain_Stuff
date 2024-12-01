from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor, AgentType, Tool
from langchain_experimental.tools import PythonREPLTool # It's a python shell that can execute python commands and python code
from langchain_experimental.agents import create_csv_agent
from langchain import hub

load_dotenv()


def main():
    print('Start...')

    instructions = """ 
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can sue to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running the code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return 'I don't know' as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()] # It's a python shell that can execute python commands and python code
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOllama(model='mistral'),
        tools=tools
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # python_agent_executor.invoke(
    #     input={
    #         "input":"""generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    csv_agent_executor:AgentExecutor= create_csv_agent(llm =ChatOllama(model='mistral'), 
                                 path = 'episode_info.csv', 
                                 verbose = True,
                                 allow_dangerous_code=True,
                                 agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    # csv_agent.invoke(input = {"input": "how many columns are ther in file episode_info.csv"})

    # csv_agent.run("print seasons ascending order of the number of episodes they have")



    ### Router Grand agent ###

    def python_agent_executor_wrapper(original_prompt: str)-> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name='Python Agent',
            func=python_agent_executor_wrapper.invoke,
            description = """useful when you need to transform natural language to python and execute the python code,
            returnin the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name='CSV Agent',
            func=csv_agent_executor.invoke,
            description = """useful when you need to answer questions over episode_info.csv,
            takes an input the entire question and returns the answer after running pandas calculations"""
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm = ChatOllama(model='mistral'),
        tools = tools
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {
                "input":"which session has the most episodes?"
            }
        ))


if __name__ == '__main__':
    main()