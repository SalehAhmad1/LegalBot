# Use the official Streamlit image as the base image
FROM python:3.10

RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV WDM_LOCAL=/app/wdm

# Copy the requirements file to the container
COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install Chrome
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    apt-get clean

# Create a non-root user
RUN useradd -m myuser

# Set permissions for the user if needed
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

RUN mkdir -p /app/cache && chmod -R 777 /app/cache
RUN mkdir -p /app/wdm && chmod -R 777 /app/wdm

# Use nohup to run Streamlit in the background and tail to monitor logs
CMD ["sh", "-c", "nohup streamlit run app.py --server.port=7860 --server.address=0.0.0.0 > nohup.out 2>&1 & tail -f nohup.out"]