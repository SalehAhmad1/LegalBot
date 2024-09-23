import requests
import json

url = "http://localhost:8080/generate"
headers = {"Content-Type": "application/json"}

# Load the request from sample_request.json or just pass the payload directly
data = {
    "text": "Explain the concept of machine learning."
}

response = requests.post(url, headers=headers, data=json.dumps(data))

# Print the generated text from the model
print(f'The response from the Saul LM 7B Instruct Model is:')
print(response.json())