import streamlit as st
import schedule
import time
import pytz
from datetime import datetime
import threading
import logging
from queue import Queue
from Scrapper.Daily_Scrapper import daily_scrapper
from RAG_v1 import RAG_Bot

@st.cache_resource
def initialize_rag_object():
    RAG_App_Object = RAG_Bot(['Uk', 'Wales', 'NothernIreland', 'Scotland'],  # Collection Names as is
                         text_splitter='SpaCy',
                         embedding_model="SentenceTransformers") 

    st.write(f'Validating the liveness of the collections...')
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
    st.write(f'Ingested into the collection {metadata["Country"]}')
    return

RAG_Object = initialize_rag_object()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Queue to pass log messages from the thread to Streamlit
log_queue = Queue()

# Function to run the scraping loop
def run_scraping_loop(log_queue):
    # Set the timezone to Pakistan Standard Time (PST)
    pakistan_timezone = pytz.timezone('Asia/Karachi')

    def job():
        # Initialize the Daily_Scrapper
        scraper = daily_scrapper(Scrapper_Folder_Path='./Scrapper')
        
        # Perform the scraping task by calling the main function
        new_content_dict = scraper.main()
        for content in new_content_dict:
            Text, MetaData = content['Text'], content['Meta Data']
            # ingest_to_rag_db(RAG_App_Object=RAG_Object, text=Text, metadata=MetaData)
            log_message = f"Scraped and Ingested a title with MetaData: {MetaData}\n\n"
            logging.info(log_message)
            log_queue.put(log_message)

    # Target time for the job
    target_time = "15:50"
    
    # Get the current time in PST
    now = datetime.now(pakistan_timezone)
    
    # Check if the current time is past the target time
    current_time_str = now.strftime("%H:%M")
    if current_time_str > target_time:
        log_message = f"Current time {current_time_str} is past the target time {target_time}. Executing job immediately.\n\n"
        logging.info(log_message)
        log_queue.put(log_message)
        job()

    # Schedule the job to run daily at 9:40 PM PST
    schedule.every().day.at(target_time).do(job)

    while True:
        now = datetime.now(pakistan_timezone)
        log_message = f"Current time: {now.strftime('%H:%M:%S')} - Waiting for {target_time}\n\n"
        logging.info(log_message)
        log_queue.put(log_message)
        schedule.run_pending()
        time.sleep(5)  # Check every minute to reduce CPU usage

# Function to display logs from the queue in the Streamlit app
def display_logs(log_queue):
    logs = []
    log_area = st.empty()  # Placeholder for log messages
    while True:
        if not log_queue.empty():
            log_message = log_queue.get()
            logs.append(log_message)
            # Update the log area with new logs
            log_area.text_area("Logs", "\n".join(logs), height=200)
        time.sleep(1)  # Refresh the display every second

# Main function to create the Streamlit app
def main():
    st.title("Daily Scraper App")
    st.write("This app will run the scraper daily at 9:40 PM Pakistan time.")

    # Run the scraping loop in a separate thread
    if st.button("Start Scraper"):
        st.write("Scraper has been started.")
        threading.Thread(target=run_scraping_loop, args=(log_queue,), daemon=True).start()
        display_logs(log_queue)

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
