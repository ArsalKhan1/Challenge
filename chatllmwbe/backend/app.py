from flask import Flask, request, jsonify, Response
from api import get_llm_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    model = data.get("model", "gpt-4")  # Default to gpt-4
    stream = data.get("stream", False)

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Check if streaming is enabled
        if stream:
            # Stream response chunks using Flask's Response generator
            def generate():
                for chunk in get_llm_response(message, model, stream=True):
                    yield chunk
            return Response(generate(), mimetype="text/plain")
        else:
            # Get the full response for non-streaming
            response_content = get_llm_response(message, model, stream=False)
            return jsonify({"response": response_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
