
# Setup of env
- create .env file 
- add your API_KEY="YOUR-API-KEY-HERE" in the file
- add your .env to .gitignore

---

# Summarize Complaint

```
    prompt_text = "The customer reported an issue with the product's battery life. They mentioned that the battery drains rapidly, even after a full charge, and lasts for less than 2 hours."
    
    result = summarize_customer_complaint(prompt_text)
    
    print(f"Generated Text: {result['generated_text']}")
    print(f"Prompt Token Count: {result['prompt_token_count']}")
    print(f"Candidates Token Count: {result['candidates_token_count']}")
    print(f"Total Token Count: {result['total_token_count']}")
```


---

# Chatbot with history


```
from chatbot import chatbot_response

if __name__ == "__main__":
    # Initialize conversation history
    conversation_history = ""

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Call the chatbot_response function with the conversation history
        result = chatbot_response(user_input, conversation_history)

        # Extract the generated response
        chatbot_reply = result['generated_text']
        print(f"Chatbot: {chatbot_reply}")

        # Append the conversation to the history (both user input and chatbot response)
        conversation_history += f"User: {user_input}\nChatbot: {chatbot_reply}\n"

        # Print token usage (optional)
        print(f"Prompt Token Count: {result['prompt_token_count']}")
        print(f"Candidates Token Count: {result['candidates_token_count']}")
        print(f"Total Token Count: {result['total_token_count']}")

```