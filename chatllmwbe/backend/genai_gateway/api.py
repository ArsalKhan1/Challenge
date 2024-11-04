import os
from openai import OpenAI
from genai_gateway_tools.oauth import OAuthConfig, OAuthTokenFetcher
from dotenv import load_dotenv
from rbc_security import enable_certs

# Load environment variables
load_dotenv(override=True)

# Enable certificates (if required)
enable_certs()

# Set up OAuth configuration
oauth_config = OAuthConfig(
    token_url=os.environ["OAUTH_TOKEN_URL"],
    client_id=os.environ["OAUTH_CLIENT_ID"],
    client_secret=os.environ["OAUTH_CLIENT_SECRET"],
    grant_type=os.environ["OAUTH_GRANT_TYPE"],
    scope=os.environ["OAUTH_SCOPE"]
)

# Fetch the OAuth token
token_fetcher = OAuthTokenFetcher(oauth_config)

# Initialize OpenAI client
client = OpenAI(
    api_key=token_fetcher.get_token(),
    base_url=os.environ.get("GW_BASE_URL")
)

def get_llm_response(message: str, model: str = "gpt-4", stream: bool = False):
    """
    Sends a request to the Generative AI Gateway and retrieves the response.

    :param message: The user input message.
    :param model: The model to use (default is "gpt-4").
    :param stream: Whether to use streaming for the response.
    :return: The response content (either a full message or streamed content).
    """
    try:
        # Create the completion request
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            stream=stream
        )

        if stream:
            # If streaming, yield each chunk of the response
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            # For non-streaming, return the full response
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error during LLM request: {e}")
        raise
