# Stage 1: Build environment using a Python base image
FROM python:3.11-slim-bullseye

# Install build tools
RUN apt-get update && apt-get install -y gcc g++ cmake zip

RUN apt install curl

# copy all code
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt 

RUN  pip install accelerate -U

RUN pip install protobuf==3.15.0

EXPOSE 8000

CMD  python -m uvicorn main:app