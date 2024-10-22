import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables.")


# Function to generate and extract the summarized response
def summarize_customer_complaint(prompt_text):
    gemini_prompt_helper = "Summarize the following customer complaint in a concise and clear manner, highlighting the main issue and any relevant details:\n"
    full_prompt = gemini_prompt_helper + prompt_text
    
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

        # Return the summarized content and token information in a dictionary
        return {
            'generated_text': generated_text,
            'prompt_token_count': prompt_token_count,
            'candidates_token_count': candidates_token_count,
            'total_token_count': total_token_count
        }
    except KeyError as e:
        raise ValueError(f"Failed to extract data: Missing key {e}")


# # Example call to the function
# if __name__ == "__main__":
#     prompt_text = "The customer reported an issue with the product's battery life. They mentioned that the battery drains rapidly, even after a full charge, and lasts for less than 2 hours."
    
#     result = summarize_customer_complaint(prompt_text)
    
#     print(f"Generated Text: {result['generated_text']}")
#     print(f"Prompt Token Count: {result['prompt_token_count']}")
#     print(f"Candidates Token Count: {result['candidates_token_count']}")
#     print(f"Total Token Count: {result['total_token_count']}")