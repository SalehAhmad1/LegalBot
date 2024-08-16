import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Union, List

from Database.Database_Weaviate import Database_Weaviate
from LLM.LLM_GGUF import LLM_GGUF

from langchain_weaviate.vectorstores import WeaviateVectorStore

class RAG_Bot:
    def __init__(self, collection_names=['Uk', 'Wales', 'NothernIreland', 'Scotland'], text_splitter='SpaCy', embedding_model="SentenceTransformers"):
        """
        Initializes the RAG_Bot object.
        
        Args:
            collection_names (list, optional): A list of collection names. Defaults to ['Uk', 'Wales', 'Nothernireland', 'Scotland'].
        """
        self.vector_db = Database_Weaviate(collection_names=collection_names, text_splitter=text_splitter, embedding_model=embedding_model)
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

    def __collection_routing(self, query) -> Union[str, List[str], None]:
        """
        This function will route the query to the correct collection.
        
        Args:
            query (str): The query to be routed.

        Returns:
            Union[str, None]: The collection name or None if no collection matches the query.
        """

        def check_for_existence_of_collection_names(query:str, collection_names:List[str]=['Uk', 'Wales', 'Nothernireland', 'Scotland']) -> Union[str, None]:
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

    def query(self, query:str, k:int=1, search_type='Hybrid'):
        """
        Performs a RAG query on the specified collection using the Saul LLM

        Args:
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.

        Returns:
            Response: The response from the LLM.
        """

        Collection_to_query_from = self.__collection_routing(query)
        print(f'Collection_to_query_from: {Collection_to_query_from}')

        if not isinstance(Collection_to_query_from, list) and Collection_to_query_from == None:
            print('There was no collection mentioned in the query. Kindly mention a collection name/s for the query to be executed.')

        elif isinstance(Collection_to_query_from, list):
            self.__query_all(query=query, k=k, collection_names=Collection_to_query_from, search_type=search_type)

    def __query_one(self, collection_name, query, k=1):
        """
        Performs a RAG query on the specified collection using the Saul LLM.
        
        Args:
            collection_name (str): The name of the collection in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        
        #Validate existence of the collection itself first.
        Validity = self.is_collection_empty(collection_name)
        print(f'The Collection: {collection_name} is empty(0)/Not Empty(1): {Validity}')
        
        if not Validity:
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
        
    def __query_all(self, query, k=1, collection_names:List[str]=['Uk', 'Wales', 'Nothernireland', 'Scotland'], search_type='Hybrid'):
        """
        Performs a RAG query on multiple specified collections using the Saul LLM.
        
        Args:
            collection_name (str): The name of the collection in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 1.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        
        # Creating a WeaviateVectorStore for all counties one by one
        for collection_name in collection_names:
            #Validate existence of the collection itself first.
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
                    print(f'The response is from the collection: {collection_name}')
                    print(response)
                    print('-')

                elif search_type == 'Hybrid':
                    current_collection = self.vector_db.client.collections.get(collection_name)
                    responses = current_collection.query.hybrid(query=query,
                                                                vector=self.vector_db.embeddings.embed_query(query),
                                                                limit=k)
                    Text_Docs = []
                    Text_Meta_Datas = []
                    
                    for o in responses.objects: #output docs of the hybrid search
                        Text_Docs.append(o.properties['text'])
                        Text_Meta_Datas.append({k: v for k, v in o.properties.items() if k != 'text'})

                    def format_docs(docs):
                        print(f'The retrieved documents are:')
                        for idx,(doc,meta) in enumerate(zip(docs,Text_Meta_Datas)):
                            print(f'{idx} - Content: {doc[:50]}... - MetaData: {meta}')
                        return "\n\n".join(doc for doc in docs)

                    concat_docs = format_docs(Text_Docs)

                    response = self.llm.chat(context={concat_docs},
                                            query={query},
                                            max_new_tokens=250)
                    print(f'The response is from the collection: {collection_name}')
                    print(response)
                    print('-')
                    
    def is_collection_empty(self, collection_name: str) -> bool:
        current_client = self.vector_db.client.collections.get(collection_name)
        for doc in current_client.iterator():
            return False
        return True

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
                is_empty = self.is_collection_empty(collection)
                if not is_empty:
                    self.get_list_of_all_docs(collection)

        elif isinstance(collection_name, str):
            collection_existence_validity = self.is_collection_empty(collection_name)
            if not collection_existence_validity:
                print(f'The collection {collection_name} has the following documents:')
                current_client = self.vector_db.client.collections.get(collection_name)
                for item in current_client.iterator():
                    for idxKey,Key in enumerate(item.properties.keys()):
                        print(f'{Key}:  {item.properties[Key]}')
                print('\n\n')