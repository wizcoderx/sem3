import os
import requests
import json
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from video_text import analyze_video
from train_model import classify_image
# Load environment variables (for API key)
load_dotenv()
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables.")

app = Flask(__name__)

# Function to classify and summarize the issue using Gemini API
def summarize_issue(description, classification_output):
    gemini_prompt_helper = """
    You are a railway digital helper means you will get a problem and you will give answer in a solution provider way to the railway management hence your job is to Summarize the following information in a short-concise detailed, comprehensive manner and in one paragraph.
    Please include all relevant aspects of the issue and provide a thorough explanation of the situation.
    Ensure the summary must cover any potential causes, consequences in one paragraph, and provide possible solutions in bullet points.
    You have a high understanding of different departments of Indian railway. More precisely the understanding of security, electricity, and water issue-related departments.

    Issue classification: {classification_output}
    User description: {description}

    Provide a comprehensive summary below:
    """

    full_prompt = gemini_prompt_helper.format(classification_output=classification_output, description=description)

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
            'problem category': classification_output,
            'summary': generated_text,
            'prompt_token_count': prompt_token_count,
            'candidates_token_count': candidates_token_count,
            'total_token_count': total_token_count
        }
    except KeyError as e:
        raise ValueError(f"Failed to extract data: Missing key {e}")

# Log token usage to a file with date and time
def log_token_usage(prompt_token_count, candidates_token_count, total_token_count):
    log_data = {
        'date': str(datetime.now()),
        'prompt_token_count': prompt_token_count,
        'candidates_token_count': candidates_token_count,
        'total_token_count': total_token_count
    }
    with open('token_usage.log', 'a') as log_file:
        log_file.write(json.dumps(log_data) + '\n')

@app.route('/classify', methods=['GET'])
def classify_image_route():
    image_url = request.args.get('image_url')
    app.logger.debug(f'Received image_url: {image_url}')

    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400

    # Here, call the image classification function from your backend (not provided in your code).
    # For demonstration purposes, we'll simulate the classification output.

    # Simulate classification logic - replace this with the actual call to your classification function
    classification_output = classify_image(image_url)  # Assume classify_image() returns a string

    # Use a fixed user description for summarization (this can be changed as needed)
    user_description = f"An issue detected is classified as : {classification_output} Two people fighting on inside the train coach. " # here you need to cheangge the desciption and accept the one that will be entered by the user.
    print(classification_output)
    # Generate the summary using the Gemini API
    summary_result = summarize_issue(user_description, classification_output)

    # print(summary_result)

    # Log token usage
    log_token_usage(
        summary_result['prompt_token_count'],
        summary_result['candidates_token_count'],
        summary_result['total_token_count']
    )

    # Return the result as a JSON response
    return jsonify(summary_result)

if __name__ == "__main__":
    app.run(debug=True)


# http://localhost:5000/classify?video_url=https%3A%2F%2Futfs.io%2Ff%2F978f9838-8ab3-4c8c-a1c8-efb41f609597-b08w7r.mp4
# http://localhost:5000/image?image_url=https://utfs.io/f/059547f8-fafc-482b-9b53-5e5dbefbc2e4-m2318w.jpeg