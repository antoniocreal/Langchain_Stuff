import os 

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOllama

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print('Retrieving...')

    embeddings = OllamaEmbeddings(model="llama3")
    llm = ChatOllama(model = 'mistral')

    query = 'What is Pinecone in machine learning?'
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input = {})
    # print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings) #The document we want to give as context to the LLM was previously ingested in this vector store

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) # The chain takes a list of documents and formats them all into a prompt
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke(input={"input": query})

    # print(result)

    template = """Use the following pices of context to answer the question at the end. 
    If you don't know the answer, just say you don't know, don't try to make up and answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thank's for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)


