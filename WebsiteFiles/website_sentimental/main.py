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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)

# Allow only requests from the frontend
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

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
    app.run(port=7423, debug=True)

