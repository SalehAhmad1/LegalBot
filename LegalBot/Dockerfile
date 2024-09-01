# Use the official Streamlit image as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

ENV TRANSFORMERS_CACHE=/app/cache/transformers

# Copy the requirements file to the container
COPY . /app

RUN mkdir -p /app/cache && chmod -R 777 /app/cache

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install Chrome
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    apt-get clean

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]