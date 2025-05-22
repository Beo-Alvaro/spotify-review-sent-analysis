#!/usr/bin/env python3
"""
This script rebuilds the model files to be compatible with the current numpy version.
Run this locally to fix the 'numpy._core' error in the Docker container.
"""

import sys
import joblib
import numpy
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_model(model_path):
    """Load and re-save a model with the current numpy version"""
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Model loaded successfully. Re-saving with numpy {numpy.__version__}")
        new_path = f"{os.path.splitext(model_path)[0]}_new.pkl"
        joblib.dump(model, new_path)
        
        logger.info(f"Model saved to {new_path}")
        return new_path
    except Exception as e:
        logger.error(f"Failed to rebuild model: {e}")
        return None

def main():
    logger.info(f"Using numpy version: {numpy.__version__}")
    
    # Try to rebuild the model files
    model_path = "sentiment_model.pkl"
    vectorizer_path = "tfidvectorizer.pkl"
    
    rebuilt_model = rebuild_model(model_path)
    rebuilt_vectorizer = rebuild_model(vectorizer_path)
    
    if rebuilt_model and rebuilt_vectorizer:
        logger.info("Both model files rebuilt successfully!")
        logger.info(f"1. Replace {model_path} with {os.path.basename(rebuilt_model)}")
        logger.info(f"2. Replace {vectorizer_path} with {os.path.basename(rebuilt_vectorizer)}")
    else:
        logger.error("Failed to rebuild model files.")
        logger.info("Alternative solution: Recreate the models from scratch with this numpy version")

if __name__ == "__main__":
    main() 