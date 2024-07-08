import weaviate
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List

import warnings
warnings.filterwarnings("ignore")

import tempfile

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class WeaviateDB():
    def __init__(self, collection_names: List[str]=['Uk', 'Whales', 'NothernIreland', 'Scotland']):
        """
        Initializes a WeaviateDB object.

        Args:
            collection_names (List[str], optional): A list of collection names. Defaults to ['Uk', 'Whales', 'NothernIreland', 'Scotland'].

        Initializes the following attributes:
            - collections (List[str]): A list of collection names.
            - embeddings (HuggingFaceInferenceAPIEmbeddings): An instance of HuggingFaceInferenceAPIEmbeddings.
            - clients (dict): A dictionary of clients, where the keys are collection names and the values are Weaviate clients.
            - vector_stores (dict): A dictionary of vector stores, where the keys are collection names and the values are WeaviateVectorStore instances.

        """
        if collection_names is None:
            collection_names: str=['Uk', 'Whales', 'NothernIreland', 'Scotland']
        
        self.collections = collection_names
        self.embeddings = self.__initialize_embeddings()
        self.clients = self.__initialize_clients()
        self.vector_stores = self.__initialize_vector_stores()
        
    def validate_collection(self):
        """
        Validates the status of each collection in the WeaviateDB object.
        Raises an exception if a cluster is not live.
        """
        for collection_name in self.collections:
            if not self.clients[collection_name].is_live():
                raise Exception(f'Cluster {collection_name} is not live')
            else:
                print(f'Validating collection: {collection_name} - Cluster Status:{self.clients[collection_name].is_live()}')
            
    def __initialize_clients(self) -> dict:
        """
        Initializes a dictionary of Weaviate clients for each collection in the WeaviateDB object.
        
        Returns:
            dict: A dictionary where the keys are collection names and the values are Weaviate clients.
        """
        clients = {}
        for name in self.collections:
            clients[name] = weaviate.connect_to_wcs(
                cluster_url=os.environ.get(f"WEAVIATE_URL_{name.upper()}"),
                auth_credentials=weaviate.AuthApiKey(api_key=os.environ.get(f"WEAVIATE_API_KEY_{name.upper()}")),
            )
        return clients

    def __initialize_vector_stores(self) -> dict:
        """
        Initializes a dictionary of WeaviateVectorStore objects for each collection in the WeaviateDB object.
        
        Returns:
            dict: A dictionary where the keys are collection names and the values are WeaviateVectorStore objects.
        """
        vector_stores = {}
        for name in self.collections:
            vector_stores[name] = WeaviateVectorStore(
                client=self.clients[name],
                index_name=name,
                text_key="question",
                embedding=self.embeddings
            )
        return vector_stores
    
    def __initialize_embeddings(self) -> HuggingFaceInferenceAPIEmbeddings:
        """
        Initializes the embeddings for the WeaviateDB object.

        Returns:
            HuggingFaceInferenceAPIEmbeddings: An instance of HuggingFaceInferenceAPIEmbeddings with the specified model name and API key.
        """
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv('HF_PASS'),
            model_name="Equall/Saul-7B-Base"
        )
        return embeddings
    
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
        current_db = self.vector_stores[collection_name]
        
        def create_temp_txt_file(text):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            with open(temp_file.name, 'w') as f:
                f.write(text)
                f.seek(0)
                return temp_file
            
        temp_txt_file = create_temp_txt_file(text)
        loader = TextLoader(temp_txt_file.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        ids = current_db.add_documents(documents=docs)
        print(f'File with data {text} added to {collection_name} with ids: {ids}')
        
    def test(self, collection_name, query, k=3):
        """
        Performs a similarity search on the specified collection in the Weaviate database using the given query.
        
        Args:
            collection_name (str): The name of the collection in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 3.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        current_db = self.vector_stores[collection_name]
        docs = current_db.similarity_search_with_score(f"{query}", k=k)
        for doc in docs:
            print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")
            
    def empty_collection(self, collection_name: str) -> None:
        """
        A function to empty a specified collection in the Weaviate database.

        Parameters:
            collection_name (str): The name of the collection in the database.

        Returns:
            None
        """
        current_db = self.vector_stores[collection_name]
        current_db.empty()
        
# if __name__ == "__main__":
#     vector_db = WeaviateDB(
#         collection_names=['Uk', 'Whales']
#     )
#     # vector_db.validate_collection()
    
#     # vector_db.add_text_to_db(
#     #     collection_name='Whales',
#     #     text='saleh is such a good and handsome boy. I cannot tell you enough',
#     #     metadata={'question': 'hello'}
#     # )
    
#     # vector_db.test(
#     #     collection_name='Whales',
#     #     query="Saleh Ahmad"
#     # )