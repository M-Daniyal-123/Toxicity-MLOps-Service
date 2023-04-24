from flask import Flask, request, jsonify
import boto3
import os
from src.clean_text import preprocess_text
from transformers import AutoTokenizer
import src.config as config
import torch
import torch.nn as nn
import numpy as np

from model import ToxicityModel

## Model path
production_dir = "production"
model_name = "model.bin"


## Downloading the Model from s3 bucket
def download_model():
    global production_dir
    global model_name

    ## Trying to creating production model directory
    try:
        os.mkdir(production_dir)
    except:
        pass

    ## Download if only model doesnot exists
    if not os.path.exists(os.path.join(production_dir, model_name)):
        bucket_name = "toxic-comments19032023"  ## Bucket name
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        )

        ## Connecting to s3 client
        s3 = session.client("s3")
        s3.download_file(
            bucket_name, "best_weights", os.path.join(production_dir, model_name)
        )

        print("Model downloaded from s3 bucket")
    else:
        print("Model already exists")


## Calling model download
download_model()

## loading the model
model = torch.load(
    os.path.join(production_dir, model_name), map_location=torch.device("cpu")
)

## Laoding the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path, do_lower=True)

## Setting up flask application
app = Flask(__name__)


## setting up flask decorator
@app.route("/predict", methods=["POST"])
def classify_text():
    ## Checking if text key exists
    if request.json.get("text", None):
        text = request.json["text"]
        if text == "":
            return {"response": "No Text found"}

        ## Applying some text cleaning
        text = str(preprocess_text(text))

        ## tokenizing the text
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )

        out = model(**tokens)
        result = out[0].detach().numpy()

        ## Presenting the response as key value pair
        response = {}
        for lb, rs in zip(config.labels, result):
            response[lb] = rs.astype(float)

        print(response)
        return {"response": response}

    else:
        return {"response": "No text key"}


if __name__ == "__main__":
    app.run(port=5001)
