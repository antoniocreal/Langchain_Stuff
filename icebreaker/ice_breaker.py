# import os

# if '__name__':
#     print('Hello Langchain!')
#     print(os.environ['OPENAI_API_KEY'])

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile



if __name__ == "__main__":
    load_dotenv()

    print('Hello Langchain')

    information = " Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. He is the wealthiest individual in the world; as of November 2024 Forbes estimates his net worth to be US$304 billion"



    summary_template = """
    given the information {information} about a person I want you to create:
    1: short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ['information'], template = summary_template
    )

    llm = ChatOllama(model="mistral",temperature=0)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")



    chain = summary_prompt_template | llm | StrOutputParser()
    # res = chain.invoke(input={"information": information})
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
    )
    res = chain.invoke(input={"information": linkedin_data})
    print(res)