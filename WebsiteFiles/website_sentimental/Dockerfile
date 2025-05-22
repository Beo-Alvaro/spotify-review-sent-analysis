FROM python:3.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies with exact versions
RUN pip install numpy==1.16.6 \
    scikit-learn==0.24.2 \
    joblib==1.0.1 \
    flask==2.0.3 \
    flask-cors==3.0.10 \
    nltk==3.6.7 \
    pandas==1.3.5 \
    gunicorn==20.1.0

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy application files
COPY . /app/

# Show current directory contents for debug
RUN ls -la /app/

# Make port available
EXPOSE $PORT

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT main:app 