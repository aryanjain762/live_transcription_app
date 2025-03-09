from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
from textblob import TextBlob
import spacy
from collections import Counter
import re

app = Flask(__name__)
CORS(app)

# Initialize Groq client
client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

nlp = spacy.load('en_core_web_sm')

@app.route('/api/analyze', methods=['POST'])
def analyze_transcription():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        transcription = data.get('transcription', [])
        if not transcription:
            return jsonify({'error': 'Empty transcription provided'}), 400

        # Combine transcription text
        full_text = ' '.join([item['text'] for item in transcription])

        # If it's a metrics analysis request
        if data.get('analysisType') == 'metrics':
            # Perform sentiment analysis
            sentiment = analyze_sentiment(full_text)
            
            # Perform intent detection
            intent = detect_intent(full_text)
            
            # Extract topics and keywords
            topic, keywords = extract_topics_and_keywords(full_text)
            
            return jsonify({
                'sentiment': sentiment,
                'intent': intent,
                'topic': topic,
                'keywords': keywords
            })

        # If it's a question-answering request
        question = data.get('question')
        transcription_type = data.get('type')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Create prompt based on transcription type
        type_prompts = {
            'medical': "As a medical transcription analyst, review this medical transcript and answer the following question: ",
            'legal': "As a legal transcription analyst, review this legal document and answer the following question: ",
            'business': "As a business analyst, review this meeting transcript and answer the following question: ",
            'academic': "As an academic researcher, review this lecture transcript and answer the following question: ",
            'general': "Review this transcript and answer the following question: "
        }

        prompt = f"""{type_prompts.get(transcription_type, type_prompts['general'])}

Transcript:
{full_text}

Question: {question}

Please provide a detailed and professional analysis based on the transcript content."""

        # Call Groq API
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a professional transcription analyst specializing in various fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return jsonify({
            'answer': completion.choices[0].message.content
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Convert polarity to a score between -1 and 1
    score = analysis.sentiment.polarity
    
    # Determine sentiment label
    if score > 0.3:
        label = "Positive"
    elif score < -0.3:
        label = "Negative"
    else:
        label = "Neutral"
    
    return {
        "score": score,
        "label": label
    }

def detect_intent(text):
    # Simple rule-based intent detection
    intents = {
        r'\b(how|what|why|when|where|who)\b': 'Question/Inquiry',
        r'\b(must|should|need|have to)\b': 'Requirement/Obligation',
        r'\b(can|could|would|will)\b': 'Request/Possibility',
        r'\b(thank|thanks|appreciate)\b': 'Gratitude',
        r'\b(problem|issue|error)\b': 'Problem Report'
    }
    
    for pattern, intent in intents.items():
        if re.search(pattern, text.lower()):
            return intent
    
    return "General Statement"

def extract_topics_and_keywords(text):
    doc = nlp(text)
    
    # Extract nouns and noun phrases as potential topics/keywords
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Count frequencies
    all_terms = nouns + [np for np in noun_phrases if len(np.split()) > 1]
    term_frequencies = Counter(all_terms)
    
    # Get the most common topic (most frequent term)
    topic = term_frequencies.most_common(1)[0][0] if term_frequencies else "General"
    
    # Get top keywords
    keywords = [term for term, freq in term_frequencies.most_common(5)]
    
    return topic, keywords

if __name__ == '__main__':
    if not os.getenv('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY environment variable is not set!")
    # Add this line to run the server
    app.run(debug=True, host='0.0.0.0', port=5000)
