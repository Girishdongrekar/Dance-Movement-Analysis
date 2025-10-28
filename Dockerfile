# Base image
FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY app /app/app
COPY streamlit_app.py /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000 8501

# Run both FastAPI and Streamlit
CMD sh -c "\
echo 'Starting FastAPI on http://0.0.0.0:8000' && \
uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
echo 'Streamlit UI → http://localhost:8501' && \
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"
