FROM python:3.8-slim

WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install numpy explicitly with the exact version that should work
RUN pip install numpy==1.22.4

# Copy specific dependencies with compatible versions
RUN pip install scikit-learn==1.0.2 \
    joblib==1.1.1 \
    flask==2.2.5 \
    flask-cors==3.0.10 \
    nltk==3.8.1 \
    pandas==1.5.3 \
    gunicorn==21.2.0

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy application files
COPY . /app/

# Show current directory contents for debug
RUN ls -la /app/

# Make port available to the world outside this container
EXPOSE $PORT

# Run the application when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT main:app 