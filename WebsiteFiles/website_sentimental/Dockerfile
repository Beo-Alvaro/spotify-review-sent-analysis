FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install numpy explicitly first (crucial for fixing numpy._core error)
RUN pip install numpy==1.23.5

# Copy requirements and install remaining dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy all other files
COPY . /app/

# Show current directory contents for debug
RUN ls -la /app/

# Make port 8000 available to the world outside this container
EXPOSE $PORT

# Run the application when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT main:app 