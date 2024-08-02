import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Union, List

from Database.Database_Weaviate import Database_Weaviate
from LLM.LLM_GGUF import LLM_GGUF

from langchain_weaviate.vectorstores import WeaviateVectorStore

class RAG_Bot:
    def __init__(self, collection_names=['Uk', 'Wales', 'Nothernireland', 'Scotland']):
        """
        Initializes the RAG_Bot object.
        
        Args:
            collection_names (list, optional): A list of collection names. Defaults to ['Uk'].
        """
        self.vector_db = Database_Weaviate(collection_names=collection_names)
        self.llm = LLM_GGUF()

    def add_text(self, collection_name, text, metadata=None):
        """
        Adds text data to a specified collection in the Weaviate database.
        
        Args:
            collection_name (str): The name of the collection in the database.
            text (str): The text data to be added.
            metadata (dict): Additional metadata associated with the text.
        """
        self.vector_db.vector_store = WeaviateVectorStore(
            client=self.vector_db.client,
            index_name=collection_name,
            text_key="text",
            embedding=self.vector_db.embeddings,
        )
        
        self.vector_db.add_text_to_db(
            collection_name=collection_name,
            text=text,
            metadata=metadata
        )

    def query(self, collection_name, query, k=1):
        """
        Performs a RAG query on the specified collection using the Ollama LLM.
        
        Args:
            collection_name (str): The name of the collection in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        
        # Creating a WeaviateVectorStore at runtime because we don't know the collection name beforehand
        self.vector_db.vector_store = WeaviateVectorStore(
            client=self.vector_db.client,
            index_name=collection_name,
            text_key="text",
            embedding=self.vector_db.embeddings,
        )
        
        # Get the current WeaviateVectorStore
        current_db = self.vector_db.vector_store
        
        # Create a retriever for the current database
        retriever = current_db.as_retriever(
            search_kwargs={"k": k})

        # Function to format documents into a single context string
        def format_docs(docs):
            print(f'The retrieved documents are:')
            for idx,doc in enumerate(docs):
                print(f'{idx} - Content: {doc.page_content[:50]}... - MetaData: {doc.metadata}')
            return "\n\n".join(doc.page_content for doc in docs)
        
        retrieved_docs = retriever.get_relevant_documents(query)
        context = format_docs(retrieved_docs)
        
        response = self.llm.chat(context={context},
                                query={query},
                                max_new_tokens=250)
        print('-')
        print(response)

    def get_list_of_all_docs(self, collection_name:Union[str, List[str]]=None) -> None:
        """
        Function to get the list of all documents in the specified collection.
        
        Args:
            collection_name (Union[str, List[str]], optional): The name of the collection in the database. Defaults to None.
        
        Returns:
            None
        """
        if isinstance(collection_name, list):
            for collection in collection_name:
                self.get_list_of_all_docs(collection)

        elif isinstance(collection_name, str):
            print(f'The collection {collection_name} has the following documents:')
            current_client = self.vector_db.client.collections.get(collection_name)
            for item in current_client.iterator():
                for idxKey,Key in enumerate(item.properties.keys()):
                    print(f'{Key}:  {item.properties[Key]}')
            print('\n\n')
                      
# if __name__ == '__main__':
#     collection_names = ['Uk', 'Wales', 'Nothernireland', 'Scotland']
#     bot = RAG_Bot(collection_names=collection_names)
#     bot.add_text(collection_name='Wales', text='Wally', metadata={'name': 'saul'})
#     bot.add_text(collection_name='Scotland', text='Scotty', metadata={'name': 'saul'})
#     bot.get_list_of_all_docs(collection_name='Wales')
#     bot.get_list_of_all_docs(collection_name='Scotland')
#     bot.query(collection_name='Wales', query='Wally')
#     bot.query(collection_name='Scotland', query='wally is what?')