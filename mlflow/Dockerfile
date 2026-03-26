FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Now copy project files
COPY . .
ENV PYTHONPATH=/app/src
CMD ["python3", "app.py"]