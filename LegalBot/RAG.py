from langchain_community.llms import Ollama
from VectorDataBase import WeaviateDB
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGDemo:
    def __init__(self, collection_names=['Uk']):
        """
        Initializes the RAGDemo object.
        
        Args:
            collection_names (list, optional): A list of collection names. Defaults to ['Uk'].
        """
        self.vector_db = WeaviateDB(collection_names=collection_names)

    def add_text(self, collection_name, text, metadata):
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

    def query(self, collection_name, query, k=3):
        """
        Performs a RAG query on the specified collection using the Ollama LLM.
        
        Args:
            collection_name (str): The name of the collection in the database.
            query (str): The query to search for similar documents.
            k (int, optional): The number of documents to return. Defaults to 3.
        
        Returns:
            None
        
        Prints the similarity score and the content of the top k documents that match the query.
        """
        current_db = self.vector_db.vector_stores[collection_name]
        
        # Create a retriever for the current database
        retriever = current_db.as_retriever()

        # Initialize Ollama LLM
        llm = Ollama(model="llama3")

        # Define the prompt template
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        # Create a chat prompt template from the template
        prompt = ChatPromptTemplate.from_template(template)

        # Function to format documents into a single context string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Execute the chain with the query and print the response
        for word in rag_chain.stream(input=f"{query}"):
            print(word, end="")
        print()

# Example usage
if __name__ == "__main__":
    # Initialize the RAGDemo with the collections
    rag_demo = RAGDemo(collection_names=['Uk']) # Add other collections if needed: 'Whales', 'NothernIreland', 'Scotland'
    
    # Add a sample text to the 'Uk' collection
    rag_demo.add_text(
        collection_name='Uk',
        text='Saleh has been good now.',
        metadata={'article': 15}
    )
    
    # Perform a RAG query with a sample query
    rag_demo.query(
        collection_name='Uk',
        query="Tell me about Saleh Ahmad"
    )