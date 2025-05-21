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

# Load ML model and vectorizer
def load_models():
    global model, vectorizer
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
            return False
            
        if not os.path.exists(vectorizer_path):
            logging.error(f"Vectorizer file not found at {vectorizer_path}")
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
        return False

# Load models on startup
models_loaded = load_models()

lemmatizer = WordNetLemmatizer()

def reprocess_text(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Check if models are loaded
        if not models_loaded or model is None or vectorizer is None:
            logging.error("Models not loaded properly")
            return jsonify({'error': 'Models not loaded properly. Please try again later.'}), 500

        data = request.get_json()
        logging.info(f"Received data: {data}")

        if not data or 'review' not in data:
            logging.warning("Invalid request: missing 'review' field")
            return jsonify({'error':'Invalid request: Missing review'}), 400

        review = data['review']
        processed_review = reprocess_text(review)
        
        try:
            vectorize_review = vectorizer.transform([processed_review])
            prediction = str(int(model.predict(vectorize_review)[0]))
            logging.info(f"Predicted sentiment: {prediction}")
            return jsonify({'sentiment': prediction})
        except Exception as model_error:
            logging.error(f"Error in prediction: {str(model_error)}")
            return jsonify({'error': str(model_error)}), 500

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'online',
        'models_loaded': models_loaded,
        'model_available': model is not None,
        'vectorizer_available': vectorizer is not None
    }
    return jsonify(status)

if __name__ == "__main__":
    # Use PORT environment variable with fallback to 7423
    port = int(os.environ.get("PORT", 7423))
    app.run(host='0.0.0.0', port=port, debug=True)

