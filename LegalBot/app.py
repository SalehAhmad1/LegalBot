import streamlit as st
from RAG_v1 import *

@st.cache_resource
def initialize_rag_object():
    RAG_App_Object = RAG_Bot(['Uk', 'Wales', 'NothernIreland', 'Scotland'],  # Collection Names as is
                         text_splitter='SpaCy',
                         embedding_model="SentenceTransformers") 

    st.write(f'Validating the liveness of the collections...')
    RAG_App_Object.vector_db.validate_collection()
    return RAG_App_Object

bot = initialize_rag_object()

# Streamlit interface
st.title('RAG Bot Query Interface')
st.subheader('In your input query do mention the target country i.e. one more of => Uk, Wales, Scotland, NothernIreland')

# Input field for the query
query = st.text_input('Enter your query:', 'Define labor laws and limitations within NothernIreland?')

# Parameters for the bot query
k = st.number_input('Number of results (k):', min_value=1, value=15)
search_type = st.selectbox('Search Type', ['Hybrid', 'Vector'])
multi_query = st.checkbox('Multi-query', False)
rerank = st.checkbox('Rerank', False)
verbose = st.checkbox('Verbose', False)

# Button to submit query
if st.button('Submit Query'):
    # Query the bot
    result, docs = bot.query(query=query, 
                       k=k, 
                       search_type=search_type, 
                       multi_query=multi_query, 
                       rerank=rerank, 
                       verbose=verbose, 
                       mode='eval')
    
    # Display the result
    st.write('Chatbot Response:')
    st.write(result)