FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for XGBoost and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ARG MONGO_DB_URL 

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
ENV MONGO_DB_URL=$MONGO_DB_URL

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
