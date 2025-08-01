FROM python:3.8.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --verbose -r requirements.txt