FROM continuumio/miniconda3:4.10.3

WORKDIR /app

# Create a new conda environment with compatible packages
RUN conda create -n sentiment_env python=3.7 \
    numpy=1.16.6 \
    scikit-learn=0.24.2 \
    joblib=1.0.1 \
    pandas=1.3.5 \
    flask=2.0.3 \
    nltk=3.6.7 \
    -c conda-forge -y

# Set up shell to use the new environment by default
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/sentiment_env/bin:$PATH
RUN echo "conda activate sentiment_env" >> ~/.bashrc

# Install pip packages not available in conda
RUN pip install flask-cors==3.0.10 gunicorn==20.1.0

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy application files
COPY . /app/

# Set environment variables for Flask
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=main.py

# Make port available
EXPOSE $PORT

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT main:app 