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

# Copy requirements file first
COPY requirements.txt /app/

# Install dependencies excluding numpy (already installed)
RUN grep -v "numpy" requirements.txt > requirements_no_numpy.txt && \
    pip install --no-cache-dir -r requirements_no_numpy.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy all other files
COPY . /app/

# Show current directory contents for debug
RUN ls -la /app/

# Make port available to the world outside this container
EXPOSE $PORT

# Run the application when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT main:app 