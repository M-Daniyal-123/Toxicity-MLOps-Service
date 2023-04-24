import torch
import boto3
import transformers
import flask
import numpy
import pandas
import tqdm
import datetime

with open("requirements.txt", "w") as f:
    f.write(f"boto3=={boto3.__version__}\n")
    f.write(f"transformers=={transformers.__version__}\n")
    f.write(f"Flask=={flask.__version__}\n")
    f.write(f"numpy=={numpy.__version__}\n")
    f.write(f"pandas=={pandas.__version__}\n")
    f.write(f"torch=={torch.__version__}\n")
    f.write(f"tqdm=={tqdm.__version__}\n")
