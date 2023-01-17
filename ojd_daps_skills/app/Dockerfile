# app/Dockerfile

FROM python:3.8-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    unzip \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements_app.txt

RUN python -m spacy download en_core_web_sm

ENTRYPOINT ["streamlit", "run", "app.py"]
