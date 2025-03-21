from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from collections import Counter
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize HuggingFace with token from environment variable
login(os.getenv('HUGGINGFACE_TOKEN'))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the saved spam detection model
with open('spam_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize summarization model
summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=0 if torch.cuda.is_available() else -1)

def get_important_features(text, vectorizer, classifier):
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    try:
        # Get feature importance for both classes
        if classifier.named_steps['classifier'].feature_log_prob_.shape[0] > 1:
            # Calculate difference in log probabilities between spam and ham
            coef = classifier.named_steps['classifier'].feature_log_prob_[1] - classifier.named_steps['classifier'].feature_log_prob_[0]
        else:
            coef = classifier.named_steps['classifier'].feature_log_prob_[0]
            
        # Transform the text
        text_features = vectorizer.transform([text]).toarray()[0]
        
        # Get present features and their importance
        present_features = []
        for idx, (present, feature, importance) in enumerate(zip(text_features, feature_names, coef)):
            if present > 0 and len(feature.split()) > 1:  # Only include multi-word phrases
                present_features.append({
                    'phrase': feature,
                    'importance': float(importance)
                })
        
        # Sort by absolute importance
        present_features.sort(key=lambda x: abs(x['importance']), reverse=True)
        return present_features[:5]  # Return top 5 most influential phrases
        
    except (IndexError, AttributeError) as e:
        print(f"Error extracting features: {e}")
        return []

def check_scam_indicators(text):
    scam_indicators = {
        'money_related': r'\b(money|rs|\$|cash|dollars|payment)\b',
        'urgency': r'\b(urgent|emergency|immediate|need|free)\b',
        'personal_info': r'\b(bank|account|sister|prince|princess)\b',
        'locations': r'\b(nigeria|abroad|foreign)\b',
        'crime': r'\b(jail|prison|arrest|drug|addict)\b'
    }
    
    indicators_found = {}
    for category, pattern in scam_indicators.items():
        matches = re.findall(pattern, text.lower())
        indicators_found[category] = len(matches)
    
    return indicators_found

def summarize_text(input_text):
    word_count = len(input_text.split())
    max_length = min(512, word_count // 2)
    min_length = max_length // 2

    chunks = [input_text[i:i+512] for i in range(0, len(input_text), 512)]
    summaries = [summarizer(chunk, max_length=max_length, min_length=min_length)[0]['summary_text'] for chunk in chunks]
    print(f"Summarized into {len(summaries)} chunks")
    return " ".join(summaries)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # Log complete received text
    print("\n=== RECEIVED TEXT FOR ANALYSIS ===")
    print(text)
    print("=== END OF RECEIVED TEXT ===\n")
    print(f"Text length: {len(text)} characters")
    print(f"Word count: {len(text.split())}\n")
    
    # Get vectorizer from pipeline
    vectorizer = model.named_steps['vectorizer']
    
    # Transform text using vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction using transformed text
    prediction = model.named_steps['classifier'].predict(text_vectorized)[0]
    probability = model.named_steps['classifier'].predict_proba(text_vectorized)[0]
    
    # Check for scam indicators
    scam_indicators = check_scam_indicators(text)
    indicator_count = sum(scam_indicators.values())
    
    # Adjust prediction if strong scam indicators are present
    if indicator_count >= 3:
        prediction = 1
        probability[1] = max(probability[1], 0.85)  # Increase spam probability
    
    # Get important features
    important_features = get_important_features(text, vectorizer, model)
    
    confidence = np.max(probability) * 100
    result = "Spam" if prediction == 1 else "Ham"
    
    # Log complete analysis results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nScam indicators found:", scam_indicators)
    print("\nTop influential phrases:")
    for feature in important_features:
        print(f"- {feature['phrase']}: {feature['importance']:.4f}")
    print("=== END OF ANALYSIS ===\n")
    
    # Add summarization if text is long enough (e.g., more than 100 words)
    summary = None
    if len(text.split()) >= 35:
        try:
            summary = summarize_text(text)
        except Exception as e:
            print(f"Summarization error: {e}")
            summary = None

    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 2),
        'important_features': important_features,
        'text_length': len(text),
        'word_count': len(text.split()),
        'scam_indicators': scam_indicators,
        'summary': summary
    })

@app.route('/summarize', methods=['POST'])
def summarize_route():
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        if not input_text:
            return jsonify({'error': 'No text provided'}), 400

        summary = summarize_text(input_text)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

