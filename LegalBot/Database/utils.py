import os
from dotenv import load_dotenv
Path_ENV = os.path.abspath(__file__)
Path_ENV = os.path.dirname(Path_ENV)
load_dotenv(Path_ENV+'/.env')

def get_spacy_text_splitter(chunk_size:int=2500, chunk_overlap:int=300):
    from langchain.text_splitter import SpacyTextSplitter
    text_splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap, max_length=2000000 , length_function=len)
    return text_splitter

def get_recursive_text_splitter(chunk_size:int=2500, chunk_overlap:int=300):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap, length_function=len)
    return text_splitter

def get_openai_embeddings():
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    return embeddings

def get_sentence_transformers_embeddings():
    from langchain.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings()
    return embeddings

def get_google_genai_embeddings():
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv('GEMINI_API_KEY'),
                                              model="models/embedding-001")
    return embeddings