FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install numpy explicitly with exact version (CRITICAL)
RUN pip install numpy==1.23.5

# Copy requirements file first
COPY requirements.txt /app/

# Install dependencies excluding numpy (already installed)
RUN grep -v "numpy" requirements.txt > requirements_no_numpy.txt && \
    pip install --no-cache-dir -r requirements_no_numpy.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy all other files
COPY . /app/

# Test numpy installation explicitly to verify _core is available
RUN python -c "import numpy; print('Numpy version:', numpy.__version__); import numpy._core; print('Numpy core imported successfully')"

# Show current directory contents for debug
RUN ls -la /app/

# Make port available to the world outside this container
EXPOSE $PORT

# Run the application when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT main:app 