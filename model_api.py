from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from collections import Counter

app = Flask(__name__)
CORS(app)

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def get_important_features(text, vectorizer, classifier):
    # Get feature names and their coefficients
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.named_steps['nb'].feature_log_prob_[1] - classifier.named_steps['nb'].feature_log_prob_[0]
    
    # Transform the text
    text_features = vectorizer.transform([text]).toarray()[0]
    
    # Get present features and their importance
    present_features = []
    for idx, (present, feature, importance) in enumerate(zip(text_features, feature_names, coef)):
        if present > 0:
            present_features.append({
                'word': feature,
                'importance': float(importance)
            })
    
    # Sort by absolute importance
    present_features.sort(key=lambda x: abs(x['importance']), reverse=True)
    return present_features[:5]  # Return top 5 most influential words

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
    
    # Make prediction
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]
    
    # Get important features
    vectorizer = model.named_steps['vectorizer']
    important_features = get_important_features(text, vectorizer, model)
    
    confidence = np.max(probability) * 100
    result = "Spam" if prediction == 1 else "Ham"
    
    # Log complete analysis results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nTop influential words:")
    for feature in important_features:
        print(f"- {feature['word']}: {feature['importance']:.4f}")
    print("=== END OF ANALYSIS ===\n")
    
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 2),
        'important_features': important_features,
        'text_length': len(text),
        'word_count': len(text.split())
    })

if __name__ == '__main__':
    app.run(port=5000)
