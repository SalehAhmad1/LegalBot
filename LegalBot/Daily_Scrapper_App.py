import streamlit as st
import schedule
import time
import pytz
from datetime import datetime
import threading
import logging
from Scrapper.Daily_Scrapper import daily_scrapper
from RAG_v1 import *

@st.cache_resource
def initialize_rag_object():
    RAG_App_Object = RAG_Bot(['Uk', 'Wales', 'NothernIreland', 'Scotland'], #Collection Names as is
                         text_splitter='SpaCy',
                         embedding_model="SentenceTransformers") 

    print(f'\nValidating the liveness of the collections:\n')
    RAG_App_Object.vector_db.validate_collection()
    return RAG_App_Object

def ingest_to_rag_db(RAG_App_Object, text, metadata):
    RAG_App_Object.add_text(collection_name=metadata['Country'],
                            text=text,
                            metadata={
                                'Country': metadata['Country'],
                                'LegislationType': metadata['LegislationType'],
                                'Legislation': metadata['Legislation'],
                                'Year': metadata['Year'],
                                'Title': metadata['Title']
                            }
                        )
    print(f'\nIngested into the collection {metadata["Country"]}\n')
    return


RAG_Object = initialize_rag_object()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to run the scraping loop
def run_scraping_loop():
    # Set the timezone to Pakistan Standard Time (PST)
    pakistan_timezone = pytz.timezone('Asia/Karachi')

    def job():
        # Initialize the Daily_Scrapper
        scraper = daily_scrapper()
        
        # Perform the scraping task by calling the main function
        new_content_dict = scraper.main()
        Text, MetaData = new_content_dict['Text'], new_content_dict['Meta Data']
        ingest_to_rag_db(RAG_App_Object=RAG_Object, text=Text, metadata=MetaData)
        logging.info("Scraping job executed successfully.")

    # Target time for the job
    target_time = "22:10"
    
    # Get the current time in PST
    now = datetime.now(pakistan_timezone)
    
    # Check if the current time is past the target time
    current_time_str = now.strftime("%H:%M")
    if current_time_str > target_time:
        logging.info(f"Current time {current_time_str} is past the target time {target_time}. Executing job immediately.")
        job()

    # Schedule the job to run daily at 9:40 PM PST
    schedule.every().day.at(target_time).do(job)

    while True:
        now = datetime.now(pakistan_timezone)
        logging.info(f"Current time: {now.strftime('%H:%M:%S')} - Waiting for 22:10 PM")
        schedule.run_pending()
        time.sleep(5)  # Check every minute to reduce CPU usage

# Main function to create the Streamlit app
def main():
    st.title("Daily Scraper App")
    st.write("This app will run the scraper daily at 9:40 PM Pakistan time.")

    # Run the scraping loop in a separate thread
    if st.button("Start Scraper"):
        st.write("Scraper has been started.")
        threading.Thread(target=run_scraping_loop, daemon=True).start()

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
