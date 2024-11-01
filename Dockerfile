# FROM python:3.9.20-slim
# COPY app/main.py /deploy/
# COPY app/config.yaml /deploy/
# WORKDIR /deploy/
# RUN apt update
# RUN apt install -y git
# RUN apt-get install -y libglib2.0-0
# RUN pip install --upgrade pip
# RUN pip install git+https://github.com/TheVinaySagar/Butter-.git
# EXPOSE 8080
# ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port 8080

# Use a slim Python base image
FROM python:3.9.20-slim
# Copy application code into the container
COPY app/main.py /deploy/
COPY app/config.yaml /deploy/

# Copy the local FastImageClassification package folder into the container
COPY Butterfly_Classification /deploy/FastImageClassification

# Set the working directory
WORKDIR /deploy/

# Update apt and install dependencies
RUN apt update && apt install -y git libglib2.0-0

# Install the FastImageClassification package from the local directory
RUN pip install /deploy/FastImageClassification

# Expose the port
EXPOSE 8080

# Run the application
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
