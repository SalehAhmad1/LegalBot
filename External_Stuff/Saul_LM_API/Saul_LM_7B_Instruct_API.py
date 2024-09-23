from flask import Flask, request, jsonify
import torch
from transformers import pipeline

# Initialize the text generation pipeline
pipe = pipeline("text-generation", model="Equall/Saul-Instruct-v1", torch_dtype=torch.bfloat16, device_map="auto")

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "Please provide a 'text' field in the request body"}), 400

    user_query = data['text']

    # Prepare the chat messages
    messages = [
        {"role": "user", "content": user_query},
    ]

    # Apply the chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the text
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

    # Return the generated text as a JSON response
    return jsonify({"generated_text": outputs[0]["generated_text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)