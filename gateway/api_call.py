from flask import Flask, request, jsonify
import requests
import json

# Create the Flask app
app = Flask(__name__)

# Function to handle the API call to the Generative AI Gateway
def call_gpt4_model(user_prompt):
    # Get OAuth token from the Generative AI Gateway
    oauth_token = get_oauth_token()

    # Set up the API endpoint and headers for the GPT-4 call
    api_endpoint = "https://perf-apigw-int.saifg.rbc.com/JLCO/llm-control-stack/v1/chat/completions"
    api_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {oauth_token}"
    }

    # Create the data payload for the GPT-4 model
    api_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }

    # Make the API call to the Generative AI Gateway
    response = requests.post(url=api_endpoint, headers=api_headers, data=json.dumps(api_data))
    
    # Check if the response is successful
    if response.ok:
        response_json = response.json()
        # Extract the GPT-4 response content
        return response_json["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Failed to call GPT-4 model: {response.text}")

# Function to obtain the OAuth token
def get_oauth_token():
    oauth_gen_endpoint = "https://ssoa.saifg.rbc.com:9443/as/token.oauth2"
    oauth_gen_headers = {"Content-Type": "application/x-www-form-urlencoded"}
    oauth_gen_data = {
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "grant_type": "client_credentials",
        "scope": "read"
    }

    # Make the OAuth token request
    response = requests.post(url=oauth_gen_endpoint, data=oauth_gen_data, headers=oauth_gen_headers)
    
    # Check if the token request is successful
    if response.ok:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get OAuth token: {response.text}")

# Endpoint to receive prompts from Hollama and return GPT-4 responses
@app.route('/query', methods=['POST'])
def query_gpt4():
    # Extract user input from the JSON payload
    user_input = request.json.get('user_input', '')

    # Check if user input is provided
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Call the GPT-4 model and get the response
        gpt4_response = call_gpt4_model(user_input)
        # Return the response as JSON
        return jsonify({"response": gpt4_response}), 200
    except Exception as e:
        # Return any errors that occur during the process
        return jsonify({"error": str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
