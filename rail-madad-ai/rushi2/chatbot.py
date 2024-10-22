import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables.")


# Function to generate chatbot responses based on user input and conversation history
def chatbot_response(user_input, conversation_history):
    # Concatenate the conversation history with the new user input
    chatbot_prompt_helper = "You are a helpful chatbot. Respond to the user's query in a conversational and helpful manner:\n\n"
    full_prompt = chatbot_prompt_helper + conversation_history + "\nUser: " + user_input + "\nChatbot:"

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": full_prompt
                    }
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    
    # Extracting data from the response
    try:
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        prompt_token_count = result['usageMetadata']['promptTokenCount']
        candidates_token_count = result['usageMetadata']['candidatesTokenCount']
        total_token_count = result['usageMetadata']['totalTokenCount']

        # Return the chatbot response and token information in a dictionary
        return {
            'generated_text': generated_text,
            'prompt_token_count': prompt_token_count,
            'candidates_token_count': candidates_token_count,
            'total_token_count': total_token_count
        }
    except KeyError as e:
        raise ValueError(f"Failed to extract data: Missing key {e}")




# Usage in main.py : 

# from chatbot import chatbot_response

# if __name__ == "__main__":
#     # Initialize conversation history
#     conversation_history = ""

#     while True:
#         # Get user input
#         user_input = input("You: ")

#         if user_input.lower() == "exit":
#             break

#         # Call the chatbot_response function with the conversation history
#         result = chatbot_response(user_input, conversation_history)

#         # Extract the generated response
#         chatbot_reply = result['generated_text']
#         print(f"Chatbot: {chatbot_reply}")

#         # Append the conversation to the history (both user input and chatbot response)
#         conversation_history += f"User: {user_input}\nChatbot: {chatbot_reply}\n"

#         # Print token usage (optional)
#         print(f"Prompt Token Count: {result['prompt_token_count']}")
#         print(f"Candidates Token Count: {result['candidates_token_count']}")
#         print(f"Total Token Count: {result['total_token_count']}")
