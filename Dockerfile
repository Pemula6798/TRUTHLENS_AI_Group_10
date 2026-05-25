# --- Build Backend & Final Image ---
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY web_app/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and models
# Hugging Face Docker works best when files are in a simple structure
COPY web_app/backend/ /app/
COPY saved_models/ /app/saved_models/

# Expose port (Hugging Face default)
EXPOSE 7860

# Run the application
# We point to main:app because main.py is now in /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
