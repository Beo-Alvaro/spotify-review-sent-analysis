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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)

# Allow requests from any origin for deployment flexibility
CORS(app)

logging.basicConfig(level=logging.INFO)

# Load ML model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidvectorizer.pkl')
except Exception as e:
    logging.error(f"Error loading model: {e}")

lemmatizer = WordNetLemmatizer()

def reprocess_text(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        if not data or 'review' not in data:
            logging.warning("Invalid request: missing 'review' field")
            return jsonify({'error':'Invalid request: Missing review'}), 400

        review = data['review']
        processed_review = reprocess_text(review)
        vectorize_review = vectorizer.transform([processed_review])
        prediction = str(int(model.predict(vectorize_review)[0]))

        logging.info(f"Predicted sentiment: {prediction}")
        return jsonify({'sentiment': prediction})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Use PORT environment variable with fallback to 7423
    port = int(os.environ.get("PORT", 7423))
    app.run(host='0.0.0.0', port=port, debug=True)

