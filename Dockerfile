# Use official lightweight Python image
FROM python:3.10-slim-buster

# Set working directory inside the container
WORKDIR /app

# Copy project files to /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Set environment variable to disable buffering (for clean logs)
ENV PYTHONUNBUFFERED=1

# Run Flask app from flask_app folder
CMD ["python3", "flask_app/app.py"]
