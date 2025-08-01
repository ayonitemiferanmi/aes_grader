FROM python:3.11-slim

WORKDIR /app

COPY . /app

COPY ./model/aes_grader.pkl /app/model/aes_grader.pkl


RUN pip install --upgrade pip==23.3.1

RUN pip install --no-cache-dir --progress-bar=on --verbose -r requirements.txt