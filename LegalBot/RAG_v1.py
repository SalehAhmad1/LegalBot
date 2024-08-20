import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from typing import List, Union, Optional, Dict

from dotenv import load_dotenv
Path_ENV = os.path.join(os.path.abspath(os.getcwd()), 'Database', '.env')
load_dotenv(Path_ENV)
from openai import OpenAI

from Database.Database_Weaviate import Database_Weaviate
# from LLM.LLM_GGUF import LLM_GGUF as LLM
from LLM.LLM_Ollama import LLM_Ollama as LLM

from langchain_weaviate.vectorstores import WeaviateVectorStore

class RAG_Bot:
    def __init__(self, collection_names: List[str] = ['Uk', 'Wales', 'NothernIreland', 'Scotland'], text_splitter: str = 'SpaCy', embedding_model: str = "SentenceTransformers"):
        """
        Initializes the RAG_Bot object.
        
        Args:
            collection_names (list, optional): A list of collection names. Defaults to ['Uk', 'Wales', 'NothernIreland', 'Scotland'].
            text_splitter (str, optional): The text splitter to use. Defaults to 'SpaCy'.
            embedding_model (str, optional): The embedding model to use. Defaults to "SentenceTransformers".
        """
        self.vector_db = Database_Weaviate(collection_names=collection_names, text_splitter=text_splitter, embedding_model=embedding_model)
        self.llm = LLM()

    def add_text(self, collection_name: str, text: str, metadata: Optional[Dict[str, Union[str, int]]] = None) -> None:
        """
        Adds text data to a specified collection in the Weaviate database.
        
        Args:
            collection_name (str): The name of the collection in the database.
            text (str): The text data to be added.
            metadata (Optional[dict], optional): Additional metadata associated with the text.
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

    def __collection_routing(self, query: str) -> Union[str, List[str], None]:
        """
        This function will route the query to the correct collection.
        
        Args:
            query (str): The query to be routed.

        Returns:
            Union[str, None]: The collection name or None if no collection matches the query.
        """

        def check_for_existence_of_collection_names(query: str, collection_names: List[str] = ['Uk', 'Wales', 'NothernIreland', 'Scotland']) -> Union[str, None]:
            Existing_collection_names = []
            for collection_name in collection_names:
                if collection_name.lower() in query.lower():
                    Existing_collection_names.append(collection_name)
            if len(Existing_collection_names) > 0:
                return Existing_collection_names
            else:
                return None
            
        mentioned_collections = check_for_existence_of_collection_names(query)
        if mentioned_collections == None:
            return None
        elif mentioned_collections != None and mentioned_collections != [] and len(mentioned_collections) >= 1:
            return mentioned_collections

    def query(self, query: str, k: int = 1, search_type: str = 'Hybrid', multi_query: bool = False, max_new_tokens: int = 1000) -> None:
        """
        Performs a RAG query on the specified collection using the Saul LLM

        Args:
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.
            search_type (str, optional): The type of search to perform. Defaults to 'Hybrid'.
            multi_query (bool, optional): Whether to perform multiple queries. Defaults to False.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1000.

        Returns:
            Response: The response from the LLM.
        """

        Collection_to_query_from = self.__collection_routing(query)
        print(f'Collection_to_query_from: {Collection_to_query_from}')

        if not isinstance(Collection_to_query_from, list) and Collection_to_query_from == None:
            print('There was no collection mentioned in the query. Kindly mention a collection name/s for the query to be executed.')

        elif isinstance(Collection_to_query_from, list):
            self.__query_all(query=query, k=k, collection_names=Collection_to_query_from, search_type=search_type, multi_query=multi_query, max_tokens=max_new_tokens)
        
    def __generate_multi_queries(self, query: str, k: int = 3) -> None: 
        '''
        A function that generates multiple queries for the user to answe, using the openAI GPT API Key.
        
        Args:
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 3.
        
        Returns:
            None
        ''' 
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError('OpenAI API key is not set. Please set it in the environment variables in the .env file in Database Directory.')
        
        client = OpenAI(api_key=openai_api_key)
        
        # Prompt to instruct the model to generate similar queries
        multi_query_system_prompt = f"""
        Generate {k} different but relevant variations of the following query.
        If the query contains mentions of a country, some specific legislative article numbers, names, dates etc. You donot modify/change these whatsoever. The original information should not be changed.
        The variations should maintain the same meaning but be worded differently to aid in retrieving related contexts.

        Original Query: "{query}"

        Respond with {k} new queries all in one string. Make sure to include the original query in the response as well. Your output should be one single string. No new lines, no bullet points, no other formatting.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Model specification
                messages=[
                    {"role": "system", "content": "You are a helpful legal law chatbot assistant."},
                    {"role": "user", "content": multi_query_system_prompt}
                ]
            )
            
            generated_text = response.choices[0].message.content.strip()
            generated_queries = generated_text.split("\n")
            
            # Ensure the number of generated queries matches 'k'
            if len(generated_queries) < k:
                print(f"Only {len(generated_queries)} queries were generated. Adjust the prompt to ensure generation of {k} queries.")
            
            return '\n'.join(generated_queries)
        
        except Exception as e:
            print(f"An error occurred during the OpenAI API call: {str(e)}")
            return []
    
    def __query_all(self, query: str, k: int = 1, collection_names: List[str] = ['Uk', 'Wales', 'Nothernireland', 'Scotland'], search_type: str = 'Hybrid', multi_query: bool = False, max_tokens: int = 1000) -> None:
        """
        Performs a RAG query on multiple specified collections using the Saul LLM.
        
        Args:
            collection_names (List[str]): The list of collection names in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.
            search_type (str, optional): The type of search to perform. Defaults to 'Hybrid'.
            multi_query (bool, optional): Whether to perform multiple queries. Defaults to False.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1000.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        
        # Creating a WeaviateVectorStore for all counties one by one
        for collection_name in collection_names:
            # Validate existence of the collection itself first.
            Validity = self.is_collection_empty(collection_name)
            print(f'The Collection: {collection_name} is empty(0)/Not Empty(1): {Validity}')
            
            if not Validity:
                if search_type == 'Vector':
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
                    def format_docs(docs: List[dict]) -> str:
                        print(f'The retrieved documents are:')
                        for idx, doc in enumerate(docs):
                            print(f'{idx} - Content: {doc.page_content[:50]}... - MetaData: {doc.metadata}')
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    if multi_query:
                        query = self.__generate_multi_queries(query)
                        print(f'Multi Query: {query}')
                    
                    retrieved_docs = retriever.get_relevant_documents(query)
                    context = format_docs(retrieved_docs)
                    
                    response = self.llm.chat(context={context},
                                            query={query},
                                            max_new_tokens=max_tokens)
                    print(f'\n\nThe response is from the collection: {collection_name}\n')
                    for chunk in response:
                        print(chunk, end='', flush=True)
                    print(f'\nThe response has been generated above ^\n\n')

                elif search_type == 'Hybrid':
                    current_collection = self.vector_db.client.collections.get(collection_name)
                    responses = current_collection.query.hybrid(query=query,
                                                                vector=self.vector_db.embeddings.embed_query(query),
                                                                limit=k)
                    Text_Docs = []
                    Text_Meta_Datas = []
                    
                    for o in responses.objects:  # output docs of the hybrid search
                        Text_Docs.append(o.properties['text'])
                        Text_Meta_Datas.append({k: v for k, v in o.properties.items() if k != 'text'})

                    def format_docs(docs: List[str]) -> str:
                        print(f'The retrieved documents are:')
                        for idx, (doc, meta) in enumerate(zip(docs, Text_Meta_Datas)):
                            print(f'{idx} - Content: {doc[:50]}... - MetaData: {meta}')
                        return "\n\n".join(doc for doc in docs)

                    concat_docs = format_docs(Text_Docs)
                    
                    if multi_query:
                        query = self.__generate_multi_queries(query)
                        print(f'Multi Query: {query}')

                    response = self.llm.chat(context={concat_docs},
                                            query={query},
                                            max_new_tokens=max_tokens)
                    print(f'\n\nThe response is from the collection: {collection_name}\n')
                    for chunk in response:
                        print(chunk, end='', flush=True)
                    print(f'\nThe response has been generated above ^\n\n')
                    
    def is_collection_empty(self, collection_name: str) -> bool:
        current_client = self.vector_db.client.collections.get(collection_name)
        for doc in current_client.iterator():
            return False
        return True

    def get_list_of_all_docs(self, collection_name: Union[str, List[str]] = None) -> None:
        """
        Function to get the list of all documents in the specified collection.
        
        Args:
            collection_name (Union[str, List[str]], optional): The name of the collection in the database. Defaults to None.
        
        Returns:
            None
        """
        if isinstance(collection_name, list):
            for collection in collection_name:
                is_empty = self.is_collection_empty(collection)
                if not is_empty:
                    self.get_list_of_all_docs(collection)

        elif isinstance(collection_name, str):
            collection_existence_validity = self.is_collection_empty(collection_name)
            if not collection_existence_validity:
                print(f'The collection {collection_name} has the following documents:')
                current_client = self.vector_db.client.collections.get(collection_name)
                for item in current_client.iterator():
                    for idxKey, Key in enumerate(item.properties.keys()):
                        print(f'{idxKey} - {Key}')
                        
    def delete_documents(self, collection_name: str, filter_params: Optional[Dict[str, Union[str, int]]] = None) -> None:
        """
        Deletes documents from the specified collection based on the provided filter parameters.
        
        Args:
            collection_name (str): The name of the collection in the database.
            filter_params (Optional[Dict[str, Union[str, int]]], optional): The filter parameters to identify documents for deletion. Defaults to None.
        
        Returns:
            None
        """
        current_client = self.vector_db.client.collections.get(collection_name)
        if filter_params:
            for item in current_client.iterator():
                if all(item.properties.get(k) == v for k, v in filter_params.items()):
                    current_client.delete(item.id)
                    print(f'Document with id {item.id} deleted from collection {collection_name}.')
        else:
            print("No filter parameters provided for deletion.")
