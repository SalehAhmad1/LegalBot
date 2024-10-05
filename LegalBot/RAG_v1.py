import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from typing import List, Union, Optional, Dict

from dotenv import load_dotenv
Path_ENV = os.path.join(os.path.abspath(os.getcwd()), 'Database', '.env')
load_dotenv(Path_ENV)
from openai import OpenAI

if __name__ == "__main__":
    from Database.Database_Weaviate import Database_Weaviate
    from LLM.LLM_Ollama import LLM_Ollama as LLM
else:
    from .Database.Database_Weaviate import Database_Weaviate
    from .LLM.LLM_Ollama import LLM_Ollama as LLM

from langchain_weaviate.vectorstores import WeaviateVectorStore

from ragatouille import RAGPretrainedModel

class RAG_Bot:
    def __init__(self, collection_names=['Uk', 'Wales', 'NothernIreland', 'Scotland'], text_splitter='SpaCy', embedding_model="SentenceTransformers"):
     
        self.vector_db = Database_Weaviate(collection_names=collection_names, text_splitter=text_splitter, embedding_model=embedding_model)
        self.llm = LLM()
        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def add_text(self, collection_name, text, metadata=None):
        """
        A function to add text data to a specified collection in the Weaviate database.
        
        Parameters:
            collection_name (str): The name of the collection in the database.
            text (str): The text data to be added.
            metadata (dict): Additional metadata associated with the text.
        
        Returns:
            None
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
      

        def check_for_existence_of_collection_names(query:str, collection_names:List[str]=['Uk', 'Wales', 'NothernIreland', 'Scotland']) -> Union[str, None]:
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

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def __generate_multi_queries(self, query: str, k: int = 3) -> None: 
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

    def query(self, 
              query:str,
              k:int=3, 
              search_type='Hybrid', 
              max_new_tokens=1000, 
              multi_query=False,
              rerank=False,
              verbose=False,
              mode='infer'):
        Collection_to_query_from = self.__collection_routing(query)
        print(f'Collection_to_query_from: {Collection_to_query_from}')

        if not isinstance(Collection_to_query_from, list) and Collection_to_query_from == None:
            print('There was no collection mentioned in the query. Kindly mention a collection name/s for the query to be executed.')
            return None

        elif isinstance(Collection_to_query_from, list):
            return self.__query_all(query=query, k=k,
                                    collection_names=Collection_to_query_from,
                                    search_type=search_type,
                                    max_tokens=max_new_tokens,
                                    multi_query=multi_query,
                                    rerank=rerank,
                                    verbose=verbose,
                                    mode=mode,)
        
    def __query_all(self, 
                    query,
                    k=1,
                    collection_names:List[str]=['Uk', 'Wales', 'Nothernireland', 'Scotland'],
                    search_type='Hybrid',
                    max_tokens=1000,
                    multi_query=False,
                    rerank=False,
                    verbose=False,
                    mode='infer'):
        All_Retrieved_Documents = ''
        individual_docs = []
        
        for collection_name in collection_names:
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
                    
                    # Retrieve the relevant documents
                    retrieved_docs = retriever.get_relevant_documents(query)

                    # Rerank the retrieved documents
                    if rerank:
                        context_docs_content = [doc.page_content for doc in retrieved_docs]
                        reranked_docs = self.reranker.rerank(query, context_docs_content, k=k//2 if k>10 else k)
                        
                        reranked_docs_content = []
                        for doc in reranked_docs:
                            reranked_docs_content.append(doc['content'])
                            individual_docs.append(doc['content'])
                        context = reranked_docs_content

                        if verbose:
                            print(f'The reranked retrieved documents are:')
                            for idx, doc in enumerate(reranked_docs):
                                print(f"Document {idx} - Reranked Score: {doc['score']} - MetaData: {retrieved_docs[doc['result_index']].metadata}")
                    else:    
                        context = self.format_docs(retrieved_docs)
                        if verbose:
                            print(f'The retrieved documents are:')
                            for idx, doc in enumerate(retrieved_docs):
                                individual_docs.append(doc.page_content)
                                print(f'Document {idx} - MetaData: {doc.metadata}')

                    # Add the retrieved documents to the All_Retrieved_Documents
                    All_Retrieved_Documents += f'''The following context is from the collection: {collection_name}\nThe context documents for this collection are: {context}\n'''

                elif search_type == 'Hybrid':
                    current_collection = self.vector_db.client.collections.get(collection_name)
                    responses = current_collection.query.hybrid(query=query,
                                                                vector=self.vector_db.embeddings.embed_query(query),
                                                                limit=k)
                    Text_Docs = []
                    Text_Meta_Datas = []
                    
                    for o in responses.objects: # output docs of the hybrid search
                        Text_Docs.append(o.properties['text'])
                        Text_Meta_Datas.append({k: v for k, v in o.properties.items() if k != 'text'})

                    def format_docs(docs, metas):
                        return "\n\n".join(doc for doc in docs)

                    if rerank:
                        reranked_docs = self.reranker.rerank(query, Text_Docs, k=k//2 if k>10 else k)
                        reranked_metas = []
                        reranked_texts = []
                        for doc in reranked_docs:
                            idx = doc['result_index']
                            reranked_metas.append(Text_Meta_Datas[idx])
                            reranked_texts.append(Text_Docs[idx])
                            individual_docs.append(Text_Docs[idx])
                        
                        if verbose:
                            print(f'The reranked retrieved documents are:')
                            for idx, doc in enumerate(reranked_docs):
                                print(f"Document {idx} - Reranked Score: {doc['score']} - MetaData: {reranked_metas[idx]}")

                        context = format_docs(reranked_texts, reranked_metas)
                    
                    else:
                        context = format_docs(Text_Docs, Text_Meta_Datas)
                        individual_docs = Text_Docs
                        
                    All_Retrieved_Documents += f'''The following context is from the collection: {collection_name}\nThe context documents for this collection are: {context}\n'''

        if multi_query:
            query = self.__generate_multi_queries(query=query, k=3)
            if verbose:
                print(f'Multi Query: {query}')
        
        response = self.llm.chat(context=f'{All_Retrieved_Documents}',
                                query=f'{query}',
                                max_new_tokens=max_tokens)
        
        if mode == 'infer':
            print(f'\n\nThe response is\n')
            for chunk in response:
                print(chunk, end='', flush=True)
            print(f'\nThe response has been generated above ^\n\n')
            return (None)
        elif mode == 'eval':
            output = ''
            for chunk in response:
                output += chunk
            return (output, individual_docs)
        
    def is_collection_empty(self, collection_name: str) -> bool:
        current_client = self.vector_db.client.collections.get(collection_name)
        for doc in current_client.iterator():
            return False
        return True

    def get_list_of_all_docs(self, collection_name:Union[str, List[str]]=None) -> None:
       
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