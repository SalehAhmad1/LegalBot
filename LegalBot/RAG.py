class RAG_Bot:
    def __init__(self, collection_names=['Uk', 'Wales', 'NothernIreland', 'Scotland']):
        """
        Initializes the RAG_Bot object.
        
        Args:
            collection_names (list, optional): A list of collection names. Defaults to ['Uk'].
        """
        self.vector_db = WeaviateDB(collection_names=collection_names)
        self.llm = Legal_LLM()

    def add_text(self, collection_name, text, metadata=None):
        """
        Adds text data to a specified collection in the Weaviate database.
        
        Args:
            collection_name (str): The name of the collection in the database.
            text (str): The text data to be added.
            metadata (dict): Additional metadata associated with the text.
        """
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
        current_db = self.vector_db.vector_stores[collection_name]
        
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
        if isinstance(collection_name, list):
            for collection in collection_name:
                self.get_list_of_all_docs(collection)

        elif isinstance(collection_name, str):
            print(f'The collection {collection_name} has the following documents:')
            current_client = self.vector_db.clients[collection_name].collections.get(collection_name)
            for item in current_client.iterator():
                for idxKey,Key in enumerate(item.properties.keys()):
                    print(f'{Key}:  {item.properties[Key]}')
            print('\n\n')