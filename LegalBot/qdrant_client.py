from qdrant_client import QdrantClient
from QdrantClient.models import Distance, VectorParams
from typing import List
import os

class CustomQdrantClient:
    """
    QdrantClient is a wrapper for the Qdrant vector store.
    
    Methods
    -------
    connect(host: str, port: int):
        Connects to the Qdrant server.
    upload_data(data: List[str], collection_name: str):
        Uploads data to the specified collection.
    query_data(query: str, collection_name: str, top_k: int):
        Queries the vector store and returns the top_k results.
    """

    def __init__(self):
        self.client = None
        self.api_key = os.getenv("QDRANT_API_KEY")

    def connect(self, url: str, api_key: str):
        """
        Connects to the Qdrant server.
        
        Parameters
        ----------
        host : str
            The hostname or IP address of the Qdrant server.
        port : int
            The port number of the Qdrant server.
        """
        self.client = QdrantClient(url="https://8d4daff8-fd97-4687-8ec3-ddd5b44bc6b9.us-east4-0.gcp.cloud.qdrant.io:6333",     
                                   api_key=self.api_key)

    def upload_data(self, data: List[str], collection_name: str):
        """
        Uploads data to the specified collection in the Qdrant vector store.
        
        Parameters
        ----------
        data : List[str]
            A list of strings representing the data to be uploaded.
        collection_name : str
            The name of the collection to which the data will be uploaded.
        """
        vectors = [self._vectorize(text) for text in data]
        self.client.upload_collection(collection_name=collection_name, vectors=vectors, payloads=data)

    def query_data(self, query: str, collection_name: str, top_k: int = 5) -> List[str]:
        """
        Queries the vector store and returns the top_k results.
        
        Parameters
        ----------
        query : str
            The query string.
        collection_name : str
            The name of the collection to be queried.
        top_k : int, optional
            The number of top results to return (default is 5).
        
        Returns
        -------
        List[str]
            A list of top_k results.
        """
        query_vector = self._vectorize(query)
        response = self.client.search(collection_name=collection_name, query_vector=query_vector, top_k=top_k)
        return [hit['payload'] for hit in response]

    def _vectorize(self, text: str) -> List[float]:
        """
        Converts a text string into a vector.
        
        Parameters
        ----------
        text : str
            The text string to be vectorized.
        
        Returns
        -------
        List[float]
            The vector representation of the text string.
        """
        # Dummy vectorization for demonstration purposes
        # Replace with actual vectorization logic, such as using a pre-trained model
        return [float(ord(char)) for char in text]

# Example usage
if __name__ == "__main__":
    qdrant = CustomQdrantClient()
    qdrant.connect()
    print(qdrant.client)
    
    # # Upload example data
    # example_data = ["This is a sample legal document.", "Another legal text goes here."]
    # qdrant.upload_data(data=example_data, collection_name="legal_docs")
    
    # # Query the vector store
    # query = "sample legal document"
    # results = qdrant.query_data(query=query, collection_name="legal_docs")
    # print("Query Results:", results)
