FROM python:3.8.0-slim-buster

## Updating the os
RUN apt-get update

## Creating the working directory
WORKDIR /APP

## Copy everything to the working directory
COPY . /APP

## Running upgrade
RUN pip install --upgrade pip

## Running the requirements
RUN python -m pip install --default-timeout=100 -r requirements.txt

## Exposing the port
EXPOSE 5001

## Running the app file
CMD ["python","app.py"]