if __name__ == "__main__":
    from base import Database
else:
    from .base import Database
import weaviate

import os
from dotenv import load_dotenv
Path_ENV = os.path.abspath(__file__)
Path_ENV = os.path.dirname(Path_ENV)
load_dotenv(Path_ENV+'/.env')

from typing import List

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

class Database_Weaviate(Database):
    def __init__(self, collection_names: List[str]=['Uk', 'Wales', 'NothernIreland', 'Scotland']):
        """
        Initializes a WeaviateDB object.

        Args:
            collection_names (List[str], optional): A list of collection names. Defaults to ['Uk', 'Wales', 'NothernIreland', 'Scotland'].

        Initializes the following attributes:
            - collections (List[str]): A list of collection names.
            - embeddings (SentenceTransformerEmbeddings): An instance of SentenceTransformerEmbeddings.
            - clients (dict): A dictionary of clients, where the keys are collection names and the values are Weaviate clients.
            - vector_stores (dict): A dictionary of vector stores, where the keys are collection names and the values are WeaviateVectorStore instances.

        """
        if collection_names is None:
            collection_names: str=['Uk', 'Wales', 'NothernIreland', 'Scotland']

        super().__init__()
        self.collections = collection_names
        self.embeddings = self._initialize_embeddings()
        self.__initialize_clients()
        self.__initialize_vector_stores()
        
    def validate_collection(self):
        """
        Validates the status of each collection in the WeaviateDB object.
        Raises an exception if a cluster is not live.
        """
        if self.client != None:
            print('Cluster Status:', self.client.is_live())
            print('Existing Collections:', self.client.collections.list_all(simple=True).keys())
        else:
            raise Exception('Cluster is not live/Setup')
        
            
    def __initialize_clients(self) -> None:
        """
        Initializes an object of Weaviate client which will have collections.
        
        Returns:
            None
        """
        Cluster_URL = os.getenv("URL")
        Cluster_API = os.getenv("API")
        self.client = weaviate.connect_to_wcs(
            cluster_url=Cluster_URL,
            auth_credentials=weaviate.AuthApiKey(api_key=Cluster_API),
            skip_init_checks=True,
        )
        
    def _get_client_collections(self) -> List[str]:
        """
        A function to get all the collections in the Weaviate database for all the clients.
        It first gets all the collections and then returns them as a list.

        Parameters:
            None

        Returns:
            List[str]: A list of collection names.
        """
        if self.client == None:
            raise Exception('Cluster is not live/Setup')
        else:
            return list(self.client.collections.list_all(simple=True).keys())
        
    def __verify_collections_existence_in_client(self) -> bool:
        """
        A function to verify if the collections exist in the Weaviate database for all the clients.
        
        Returns:
            bool: True if all collections exist in the Weaviate database for all the clients, False otherwise.
        """
        All_Collections = self._get_client_collections()
        for Country_Name in self.collections:
            if Country_Name not in All_Collections:
                print(f'The collection: {Country_Name} does not exist in the Weaviate Cluster hence creating it.')
                return False
        return True

    def __initialize_vector_stores(self) -> None:
        """
        Initializes a dictionary of WeaviateVectorStore objects for each collection in the WeaviateDB object.
        
        Returns:
            None
        """
        
        if isinstance(self.collections, str):
            print(f'Creating {self.collections} Weaviate Cluster - Option 1')
            self.vector_store = WeaviateVectorStore(
                client=self.client,
                index_name=self.collections, # Loads or creates if not exists
                text_key="text", #What the retrieved document actual content is in
                embedding=self.embeddings,
            )
            
        elif isinstance(self.collections, list) and len(self.collections) == 1:
            print(f'Creating {len(self.collections)} Weaviate Cluster - Option 2')
            self.vector_store = WeaviateVectorStore(
                client=self.client,
                index_name=self.collections[0], # Loads or creates if not exists
                text_key="text", #What the retrieved document actual content is in
                embedding=self.embeddings,
            )
        
        elif isinstance(self.collections, list) and len(self.collections) > 1:
            print(f'Creating {len(self.collections)} Weaviate Clusters - Option 3')
            Client_Current_Collection = self._get_client_collections()
            
            for idxCountry, Country in enumerate(self.collections):
                if Country.capitalize() not in Client_Current_Collection:
                    print(f'The collection: {Country} does not exist in the Weaviate Cluster hence creating it.')
                    self.client.collections.create(
                        name=Country,
                    )
                else:
                    print(f'The collection: {Country} already exists in the Weaviate Cluster')
            
            '''At Run time???? because langchain fucntions may not support multiple collections?'''
            self.vector_store = None
    
    def delete_collection(self, collection_name: str) -> None:
        """
        A function to delete a specified collection in the Weaviate database.

        Parameters:
            collection_name (str): The name of the collection in the database.

        Returns:
            None
        """
        if isinstance(collection_name, list):
            raise Exception('Cannot delete multiple collections at once.\nUse the delete_all_collections() method instead to delete all collections at once.')
        else:
            All_Collections = self._get_client_collections()
            if collection_name not in All_Collections:
                print(f'The collection: {collection_name} does not exist in the Weaviate Cluster hence cannot delete it.')
            else:
                print(f'The collection: {collection_name} exists in the Weaviate Cluster hence deleting it.')
                self.client.collections.delete(f'{collection_name}')
            
    def delete_all_collections(self) -> None:
        """
        A function to delete all the collections in the Weaviate database for all the clients.
        It first gets all the collections and then deletes them one by one.

        Parameters:
            None

        Returns:
            None
        """
        for Collection_Name in self._get_client_collections():
            print(f'Deleting collection: {Collection_Name}')
            self.client.collections.delete(f'{Collection_Name}')
        print(f'All collections deleted. Available collections: {self._get_client_collections()}')