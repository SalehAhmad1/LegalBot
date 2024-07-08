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
        if collection_names is None:
            collection_names: str=['Uk', 'Whales', 'NothernIreland', 'Scotland']
        
        self.collections = collection_names
        self.embeddings = self.__initialize_embeddings()
        self.clients = self.__initialize_clients()
        self.vector_stores = self.__initialize_vector_stores()
        
    def validate_collection(self):
        for collection_name in self.collections:
            if not self.clients[collection_name].is_live():
                raise Exception(f'Cluster {collection_name} is not live')
            else:
                print(f'Validating collection: {collection_name} - Cluster Status:{self.clients[collection_name].is_live()}')
            
    def __initialize_clients(self) -> dict:
        clients = {}
        for name in self.collections:
            clients[name] = weaviate.connect_to_wcs(
                cluster_url=os.environ.get(f"WEAVIATE_URL_{name.upper()}"),
                auth_credentials=weaviate.AuthApiKey(api_key=os.environ.get(f"WEAVIATE_API_KEY_{name.upper()}")),
            )
        return clients

    def __initialize_vector_stores(self) -> dict:
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
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv('HF_PASS'),
            model_name="Equall/Saul-7B-Base"
        )
        return embeddings
    
    def add_text_to_db(self, collection_name: str, text: str, metadata: dict) -> None:
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
        current_db = self.vector_stores[collection_name]
        docs = current_db.similarity_search_with_score(f"{query}", k=k)
        for doc in docs:
            print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")
        
if __name__ == "__main__":
    vector_db = WeaviateDB(
        collection_names=['Uk', 'Whales']
    )
    # vector_db.validate_collection()
    
    # vector_db.add_text_to_db(
    #     collection_name='Whales',
    #     text='saleh is such a good and handsome boy. I cannot tell you enough',
    #     metadata={'question': 'hello'}
    # )
    
    # vector_db.test(
    #     collection_name='Whales',
    #     query="Saleh Ahmad"
    # )