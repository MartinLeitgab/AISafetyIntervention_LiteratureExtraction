import openai
import os

# Make sure your API key is set correctly
openai.api_key = os.getenv("OPENAI_API_KEY")
# or
# openai.api_key = "your-api-key-here"

# Test if the key works
try:
    response = openai.models.list()
    print(f"{response}")
    print("API key is working!")
except Exception as e:
    print(f"API key issue: {e}")
