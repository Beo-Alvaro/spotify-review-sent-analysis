FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords wordnet punkt_tab

# Make port 8000 available to the world outside this container
EXPOSE $PORT

# Run the application when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT main:app 