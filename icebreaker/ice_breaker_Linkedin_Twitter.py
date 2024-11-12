from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


from third_parties.linkedin import scrape_linkedin_profile
from Agents.Linkedin_lookup_agent import lookup as linkedin_lookup_agent
from Agents.Twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets #Won't be able to import as there are no twitter tokens or API keys 


def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username)

    summary_template = """
    given the information about a person from linkedin {information} and twitter psots {twitter_posts} I want you to create:
    1. A short summary
    2. two interesting facts about them

    Use both information from linkedin and twitter
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = ChatOllama(model="mistral",temperature=0) 

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    print(res)
    print(res['text'])


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Eden Marco") # A trick is to run with Eden Marco Udemy instead of just Eden Marco

