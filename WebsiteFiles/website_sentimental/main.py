import sklearn
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import os
import sys
import random
from collections import Counter

# Download only what we need
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Explicitly allow requests from GitHub Pages and other origins
cors = CORS(app, resources={r"/*": {"origins": ["https://beo-alvaro.github.io", "http://localhost:5500", "*"]}})

logging.basicConfig(level=logging.INFO)

# Declare global variables
model = None
vectorizer = None
using_fallback = False

# Define a simple fallback sentiment analyzer using NLTK
def fallback_sentiment_analyzer(text):
    # Define basic sentiment word lists
    positive_words = ["good", "great", "excellent", "fantastic", "amazing", "wonderful", "brilliant", 
                    "love", "happy", "best", "beautiful", "enjoy", "like", "awesome", "perfect"]
    negative_words = ["bad", "terrible", "awful", "horrible", "poor", "worst", "hate", 
                    "sad", "disappointed", "unfortunate", "boring", "annoying", "dislike"]
    
    # Preprocess text
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Count sentiment words
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Determine sentiment
    if pos_count > neg_count:
        return "1"  # Positive
    elif neg_count > pos_count:
        return "0"  # Negative
    else:
        # If it's a tie or no sentiment words found, return a random sentiment
        # Slightly biased toward positive for better user experience
        return str(random.choices([0, 1], weights=[1, 1.2])[0])

# Load ML model and vectorizer
def load_models():
    global model, vectorizer, using_fallback
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths to model files
        model_path = os.path.join(current_dir, 'sentiment_model.pkl')
        vectorizer_path = os.path.join(current_dir, 'tfidvectorizer.pkl')
        
        logging.info(f"Loading model from: {model_path}")
        logging.info(f"Loading vectorizer from: {vectorizer_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            using_fallback = True
            return False
            
        if not os.path.exists(vectorizer_path):
            logging.error(f"Vectorizer file not found at {vectorizer_path}")
            using_fallback = True
            return False
            
        # Load model files
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        logging.info("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(f"Exception type: {type(e)}")
        logging.error(f"Exception traceback: {sys.exc_info()}")
        logging.info("Falling back to simple NLTK-based sentiment analysis")
        using_fallback = True
        return False

# Load models on startup
models_loaded = load_models()

# Initialize lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logging.error(f"Error initializing lemmatizer: {str(e)}")
    lemmatizer = None

# Simplified text processing that doesn't rely on punkt_tab
def reprocess_text(text):
    try:
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r"n't", " not", text)
        
        # Try to use word_tokenize if available, otherwise fall back to simple split
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logging.warning(f"word_tokenize failed: {e}, using simple split")
            tokens = text.split()
            
        # Try to use stopwords if available
        try:
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            logging.warning(f"stopwords failed: {e}, using simple stopwords")
            # Basic English stopwords
            stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                        "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                        'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
                        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                        'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
                        "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                        "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
                        "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                        'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                        'won', "won't", 'wouldn', "wouldn't"}
        
        # Filter tokens and lemmatize if possible
        if lemmatizer:
            filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        else:
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            
        return ' '.join(filtered_tokens)
    except Exception as e:
        logging.error(f"Error in text preprocessing: {str(e)}")
        # Return original text as a last resort
        return text

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        if not data or 'review' not in data:
            logging.warning("Invalid request: missing 'review' field")
            return jsonify({'error':'Invalid request: Missing review'}), 400

        review = data['review']
        processed_review = reprocess_text(review)
        
        # Check if we should use the fallback
        if using_fallback or not models_loaded or model is None or vectorizer is None:
            logging.info("Using fallback sentiment analysis")
            prediction = fallback_sentiment_analyzer(processed_review)
            return jsonify({'sentiment': prediction, 'method': 'fallback'})
        
        # Use the ML model
        try:
            vectorize_review = vectorizer.transform([processed_review])
            prediction = str(int(model.predict(vectorize_review)[0]))
            logging.info(f"Predicted sentiment: {prediction}")
            return jsonify({'sentiment': prediction, 'method': 'ml_model'})
        except Exception as model_error:
            logging.error(f"Error in prediction: {str(model_error)}")
            # Fallback to simple sentiment analysis if ML model fails
            logging.info("ML model failed, using fallback sentiment analysis")
            prediction = fallback_sentiment_analyzer(processed_review)
            return jsonify({'sentiment': prediction, 'method': 'fallback'})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        # Last resort - just return a random sentiment
        prediction = str(random.randint(0, 1))
        return jsonify({'sentiment': prediction, 'method': 'emergency_fallback'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'online',
        'models_loaded': models_loaded,
        'model_available': model is not None,
        'vectorizer_available': vectorizer is not None,
        'using_fallback': using_fallback
    }
    return jsonify(status)

if __name__ == "__main__":
    # Use PORT environment variable with fallback to 7423
    port = int(os.environ.get("PORT", 7423))
    app.run(host='0.0.0.0', port=port, debug=True)

