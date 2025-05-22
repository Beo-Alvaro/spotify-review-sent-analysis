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

# Copy application files first (so we can test with the actual models)
COPY . /app/

# Install core dependencies (excluding numpy)
RUN pip install \
    scikit-learn==0.24.2 \
    joblib==1.0.1 \
    flask==2.0.3 \
    flask-cors==3.0.10 \
    nltk==3.6.7 \
    pandas==1.3.5 \
    gunicorn==20.1.0

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Create a script to test numpy versions
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import subprocess\n\
import os\n\
\n\
# Numpy versions to try\n\
VERSIONS = ["1.19.5", "1.18.5", "1.17.5", "1.16.6", "1.20.3", "1.21.6"]\n\
\n\
def test_version(version):\n\
    print(f"\\n\\n============= Testing numpy {version} =============\\n")\n\
    # Uninstall current numpy\n\
    subprocess.call(["pip", "uninstall", "-y", "numpy"])\n\
    # Install specific version\n\
    result = subprocess.call(["pip", "install", f"numpy=={version}"])\n\
    if result != 0:\n\
        print(f"Failed to install numpy {version}")\n\
        return False\n\
    \n\
    # Test importing numpy._core\n\
    try:\n\
        import numpy\n\
        print(f"Numpy version: {numpy.__version__}")\n\
        from numpy import _core\n\
        print("Successfully imported numpy._core")\n\
        \n\
        # Try to load the model\n\
        import joblib\n\
        try:\n\
            model = joblib.load("sentiment_model.pkl")\n\
            print("Successfully loaded the model!")\n\
            return True\n\
        except Exception as e:\n\
            print(f"Failed to load model with numpy {version}: {e}")\n\
            return False\n\
    except ImportError as e:\n\
        print(f"Failed to import numpy._core with version {version}: {e}")\n\
        return False\n\
\n\
# Try each version\n\
for version in VERSIONS:\n\
    success = test_version(version)\n\
    if success:\n\
        print(f"\\n\\n************* SUCCESS: numpy {version} works! *************\\n")\n\
        # Write the working version to a file for the Dockerfile to use\n\
        with open("working_numpy_version.txt", "w") as f:\n\
            f.write(version)\n\
        sys.exit(0)\n\
\n\
print("\\n\\n************* FAILURE: No working numpy version found *************\\n")\n\
sys.exit(1)\n\
' > /app/test_numpy_versions.py

# Make the script executable
RUN chmod +x /app/test_numpy_versions.py

# Run the test script
RUN python /app/test_numpy_versions.py || echo "No compatible numpy version found"

# Install the version that worked (if found)
RUN if [ -f /app/working_numpy_version.txt ]; then \
        NUMPY_VERSION=$(cat /app/working_numpy_version.txt); \
        echo "Installing working numpy version: $NUMPY_VERSION"; \
        pip install numpy==$NUMPY_VERSION; \
    else \
        echo "No working numpy version found, using 1.19.5 as fallback"; \
        pip install numpy==1.19.5; \
    fi

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT main:app 