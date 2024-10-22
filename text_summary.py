import spacy
from transformers import pipeline

'''

This script doesn’t seem to influence app.py directly or indirectly. It iss a standalone script for text summarization and extracting relevant information from text using Spacy and a transformer model.
'''


nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization")
def extract_relevant_info(paragraph, query):
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]

    relevant_sentences = [sent for sent in sentences if query.lower() in sent.lower()]

    if relevant_sentences:
        result = ' '.join(relevant_sentences)
        summary = summarizer(result, max_length=30, min_length=10, do_sample=False)
        return summary[0]['summary_text']

    summary = summarizer(paragraph, max_length=30, min_length=10, do_sample=False)
    return summary[0]['summary_text']

paragraph = input("Enter your query.  ")
query = "electricity"
result = extract_relevant_info(paragraph, query)
print()
print()
print()
print("Response:",end=' ')
print(result)
