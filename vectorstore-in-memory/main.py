import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS #Faiss turns objects like PDF's or text files and perform similarity search
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain



if __name__ == "__main__":
    print('Reading the PDF...')
    pdf_path = '/home/acr/Documents/vectorstore-in-memory/ReAct: Synergizing Reasoning and Acting in Language Models.pdf'
    loader = PyPDFLoader(file_path=pdf_path) # Loads the document and reads it 
    documents = loader.load()  
    text_splitter = CharacterTextSplitter(chunk_size=2273, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    llm = ChatOllama(model = 'mistral')

    embeddings = OllamaEmbeddings(model = 'llama3')
    vectorstore = FAISS.from_documents(docs, embeddings) # Faiss turns the documents, chunked, and turns it into vectors and turns them into faiss. This is stored in RAM in the local machine and ends up being a vectorstore
    vectorstore.save_local("faiss_index_react")

    print('Finishe Vectorstore...')

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) # The chain takes a list of documents and formats them all into a prompt
    retrieval_chain = create_retrieval_chain(retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])













