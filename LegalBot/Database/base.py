from typing import List

if __name__ == "__main__":
    from utils import *
else:
    from .utils import *

import os
from dotenv import load_dotenv
Path_ENV = os.path.abspath(__file__)
Path_ENV = os.path.dirname(Path_ENV)
load_dotenv(Path_ENV+'/.env')

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.docstore.document import Document

class Database():
    def __init__(self, text_splitter=None, embedding_model=None) -> None:
        """
        Initializes a WeaviateDB object.
        """
        if text_splitter == None or text_splitter == 'Recursive':
            self.text_splitter = get_recursive_text_splitter()
        elif text_splitter in ['SPACY', 'spacy', 'Spacy', 'SPACy', 'spaCy', 'SpaCy']:
            self.text_splitter = get_spacy_text_splitter()
        else:
            raise Exception('Invalid Text Splitter. Choose from SpaCy or Recursive')

        if embedding_model == None or embedding_model == 'SentenceTransformers': 
            self.embeddings = get_sentence_transformers_embeddings()
        elif embedding_model in ['Openai', 'openai', 'OPENAI', 'OpenAI']:
            self.embeddings = get_openai_embeddings()
        elif embedding_model in ['Google', 'google', 'GoogleGENAI', 'gemini', 'GEMINI']:
            self.embeddings = get_google_genai_embeddings()
        else:
            raise Exception('Invalid Embedding Model. Choose from OpenAI, SentenceTransformers, or Google')
        
    def validate_collection(self):
        """
        Validates the status of each collection in the WeaviateDB object.
        Raises an exception if a cluster is not live.
        """
        raise Exception('This method should be implemented by the subclass')
            
    def _initialize_clients(self) -> None:
        """
        Initializes an object of Weaviate client which will have collections.
        
        Returns:
            None
        """
        raise Exception('This method should be implemented by the subclass')
        
    def _get_client_collections(self) -> List[str]:
        """
        A function to get all the collections in the Weaviate database for all the clients.
        It first gets all the collections and then returns them as a list.

        Parameters:
            None

        Returns:
            List[str]: A list of collection names.
        """
        raise Exception('This method should be implemented by the subclass')
        
    def _verify_collections_existence_in_client(self) -> bool:
        """
        A function to verify if the collections exist in the Weaviate database for all the clients.
        
        Returns:
            bool: True if all collections exist in the Weaviate database for all the clients, False otherwise.
        """
        raise Exception('This method should be implemented by the subclass')

    def _initialize_vector_stores(self) -> None:
        """
        Initializes a dictionary of WeaviateVectorStore objects for each collection in the WeaviateDB object.
        
        Returns:
            None
        """
        raise Exception('This method should be implemented by the subclass')
    
    def add_text_to_db(self, collection_name: str, text: str, metadata: dict) -> None:
        """
        A function to add text data to a specified collection in the Weaviate database.

        Parameters:
            collection_name (str): The name of the collection in the database.
            text (str): The text data to be added.
            metadata (dict): Additional metadata associated with the text.

        Returns:
            None
        """
        current_db = self.vector_store
        
        text_splitter = self.text_splitter
        splitted_texts = text_splitter.split_text(text)
        docs = [Document(page_content=doc, metadata=metadata) for doc in splitted_texts]
        ids = current_db.add_documents(documents=docs)
        print(f'File with data with ids: {ids}')
                        
    def delete_collection(self):
        """
        A function to delete a specified collection in the Weaviate database.

        Parameters:
            None

        Returns:
            None
        """
        raise Exception('This method should be implemented by the subclass')
            
    def delete_all_collections(self) -> None:
        """
        A function to delete all the collections in the Weaviate database for all the clients.
        It first gets all the collections and then deletes them one by one.

        Parameters:
            None

        Returns:
            None
        """
        raise Exception('This method should be implemented by the subclass')