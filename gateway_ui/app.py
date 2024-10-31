from flask import Flask, request, jsonify, stream_with_context, Response
import openai
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

app = Flask(__name__)

# Initialize OpenAI client
openai.api_key = os.getenv("OAUTH_CLIENT_SECRET")  # Make sure OAUTH_CLIENT_SECRET is your token
openai.api_base = os.getenv("GW_BASE_URL")

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    def stream_response():
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            stream=True
        )

        for chunk in response:
            if 'choices' in chunk:
                chunk_content = chunk.choices[0].delta.get('content', '')
                yield f"data: {chunk_content}\n\n"

    return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
