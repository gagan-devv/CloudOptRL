# FORCE_BUILD_1
FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY env/ ./env/
COPY app.py .
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Gradio port
EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]
