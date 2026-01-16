# 1️⃣ Base image with Python
FROM python:3.11-slim

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy your source code and models
COPY src/ ./src/
COPY models/ ./models/

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir \
    flask \
    onnxruntime \
    torch \
    torchvision \
    pillow \
    numpy \
    pandas

# 5️⃣ Expose the port Flask will run on
EXPOSE 5000

# 6️⃣ Set environment variables for Flask
ENV FLASK_APP=src/serve.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# 7️⃣ Default command to run the Flask server
CMD ["flask", "run"]
