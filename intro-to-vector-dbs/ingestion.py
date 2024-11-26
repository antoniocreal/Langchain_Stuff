import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()


if __name__ == '__main__':
    print('Ingesting')
    loader = TextLoader("/home/acr/Documents/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print('spliting...')
    textsplitter= CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0) # Each chunk size will have 1000 characters to fit the text window
    texts = textsplitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="llama3")

    print('Ingesting...')
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])

    print('Finish')


