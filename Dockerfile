# Use the official Python slim image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# 1. COPY requirements.txt and model file
# The model is copied here so it's available for the application.
COPY requirements.txt .
# Assuming your model is in the 'models/' folder relative to the root:
COPY models/ /app/models/

# 2. Pre-download the largest packages to avoid timeouts
# This is a dedicated step to handle the large TensorFlow and Keras files.
# It uses the target architecture (if defined) for optimal downloads.
RUN pip download --no-deps tensorflow keras -d /tmp/packages

# 3. RUN pip install (The Main Install Step)
# Install everything from requirements.txt, using the local cache for TF/Keras.
# The --find-links flag tells pip to look in the downloaded directory first, 
# which avoids the unreliable internet download for the huge files.
RUN pip install --no-cache-dir --find-links /tmp/packages -r requirements.txt

# 4. Download NLTK data
# This step also benefits from the cache if the installation step above is cached.
RUN python -m nltk.downloader stopwords punkt wordnet

# 5. Copy the rest of the application code
COPY . /app

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app/app_streamlit.py"]
