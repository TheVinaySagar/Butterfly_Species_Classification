FROM python:3.9.20-slim
COPY app/main.py /deploy/
COPY app/config.yaml /deploy/
WORKDIR /deploy/
RUN apt update
RUN apt install -y git
RUN apt-get install -y libglib2.0-0
RUN pip install --upgrade pip
RUN pip install git+https://github.com/TheVinaySagar/Butterfly_Classification.git
EXPOSE 8080
ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port 8080