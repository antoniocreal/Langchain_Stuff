from langchain_community.tools.tavily_search import TavilySearchResults

#Tavily Search API is a search engine optimized for LLMs, optimized for a factual, efficient, and persistent search experience
def get_profile_url_tavily(name: str):
    """Searches for Linkedin or Twitter Profile Page."""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res