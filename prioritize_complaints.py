import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables.")


# Function to prioritize customer complaints based on urgency
def prioritize_complaints(complaints_list):
    gemini_prompt_helper = "Rank the following customer complaints based on urgency, with the most urgent complaint at the top of the list:\n"
    
    # Combine the list of complaints into a formatted prompt
    complaints_text = "\n".join([f"{index + 1}. {complaint}" for index, complaint in enumerate(complaints_list)])
    full_prompt = gemini_prompt_helper + complaints_text
    
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

        # Parsing the generated text into a prioritized list
        prioritized_complaints = [complaint.strip() for complaint in generated_text.split('\n') if complaint.strip()]
        
        # Return the prioritized complaints and token information in a dictionary
        return {
            'prioritized_complaints': prioritized_complaints,
            'prompt_token_count': prompt_token_count,
            'candidates_token_count': candidates_token_count,
            'total_token_count': total_token_count
        }
    except KeyError as e:
        raise ValueError(f"Failed to extract data: Missing key {e}")


# usage
# if __name__ == "__main__":
#     complaints_list = [
#         "The customer is unable to access their account due to a login issue.",
#         "The product arrived damaged and is unusable.",
#         "The service was delayed by over an hour and the customer is unhappy with the lack of communication.",
#         "The refund for the return order has not been processed yet.",
#     ]
    
#     result = prioritize_complaints(complaints_list)
    
#     print(f"Prioritized Complaints: {result['prioritized_complaints']}")
#     print(f"Prompt Token Count: {result['prompt_token_count']}")
#     print(f"Candidates Token Count: {result['candidates_token_count']}")
#     print(f"Total Token Count: {result['total_token_count']}")
