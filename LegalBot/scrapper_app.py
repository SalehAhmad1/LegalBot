import streamlit as st
import schedule
import time
import pytz
from datetime import datetime, timedelta
import threading
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from queue import Queue
from Scrapper.Daily_Scrapper import daily_scrapper
from RAG_v1 import RAG_Bot

# Set the timezone
london_timezone = pytz.timezone('Europe/London')

# Target time for the job (e.g., 1:35 AM)
target_time = "23:58"

# Flag to indicate if the job has already run today
has_run_today = False

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
    global has_run_today

    def job():
        global has_run_today

        # Initialize the Daily_Scrapper
        scraper = daily_scrapper(Scrapper_Folder_Path='./Scrapper')
        
        # Perform the scraping task by calling the main function
        new_content_dict = scraper.main()
        for content in new_content_dict:
            f'In ingestion Loop'
            if content is None:
                break
            else:
                Text, MetaData = content['Text'], content['Meta Data']
                try:
                    ingest_to_rag_db(RAG_App_Object=RAG_Object, text=Text, metadata=MetaData)
                    log_message = f"Scraped and Ingested a title with MetaData: {MetaData}\n\n"
                    logging.info(log_message)
                    log_queue.put(log_message)
                except Exception as e:
                    log_message = f"Error when ingesting {MetaData} to Vector DB. Details: {e}\n\n"
                    logging.info(log_message)
                    log_queue.put(log_message)

        has_run_today = True  # Mark the job as done for today
    
    # Schedule the job to run daily at the specified time
    schedule.every().day.at(target_time).do(job)

    while True:
        now = datetime.now(london_timezone)
        current_time = now.strftime("%H:%M")

        # Check if the current time is past or equal to the target time
        if not has_run_today and current_time >= target_time:
            log_message = f"Current time {current_time} is past the target time {target_time}.\nExecuting job immediately.\n\n"
            logging.info(log_message)
            log_queue.put(log_message)
            job()
        else:
            log_message = f"Current time: {current_time} - Waiting for {target_time}\n\n"
            logging.info(log_message)
            log_queue.put(log_message)
        
        # Reset the flag at midnight (or the next day)
        if now.hour == 0 and now.minute == 0:
            has_run_today = False

        schedule.run_pending()
        time.sleep(50)  # Check every 50 seconds to reduce CPU usage

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

def gmail_create_draft(body='Error when ingesting new title to Vector DB. Check Logs.'):
    msg = MIMEMultipart()
    msg['From'] = "legalllm24@gmail.com"
    msg['To'] = "legalllm24@gmail.com"
    msg['Subject'] = 'Error when ingesting new title to Vector DB. Check Logs.'
    
    if not body:
        print("Warning: Email Body is empty!")
        return False
    
    body_text = body

    msg.attach(MIMEText(body_text, 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(msg['From'], 'eciu xofm laci drtc')
            text = msg.as_string()
            
            server.sendmail(msg['From'], msg['To'], text)
            print("Email sent successfully")
            return True
    except Exception as e:
        print(f"Error: unable to send email. Details: {e}")
        return False

# Main function to create the Streamlit app
def main():
    st.title("Daily Scraper App")
    st.write(f"This app will run the scraper daily at {target_time} London time.")

    # Automatically start the scraper when the app is run
    threading.Thread(target=run_scraping_loop, args=(log_queue,), daemon=True).start()
    display_logs(log_queue)

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()