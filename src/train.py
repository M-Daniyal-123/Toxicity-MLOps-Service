import mlflow
from transformers import (
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
import config
from model import ToxicityModel
from dataset import ToxicityDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import evaluate
import random
import numpy as np
import os
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train():
    ########### Setting the seed value ############
    set_seed(config.seed)

    ########### Setting the device #################
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ########### Setting up the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        config.bert_model_path, do_lower=True
    )

    ########### Setting up model name ##############
    model_name = f"model-{str(datetime.now())}"
    model_name = model_name.replace(" ", "-").replace(".", "-").replace(":", "-")
    model_name = model_name + ".bin"

    ########### Loading dataloaders
    ### train
    train_dataset = ToxicityDataset(config.train_data_path, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    ### Validation
    valid_dataset = ToxicityDataset(config.validation_data_path, tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)

    ### Setting up the model
    model = ToxicityModel(config.bert_model_path)
    model.to(device)
    print(f"Model moved to {device}")

    ########### Setting up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    total_steps = (len(train_dataloader) // config.batch_size) * config.epochs
    num_warmup_steps = total_steps // 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    ########### Setting up the loss function
    loss_fn = nn.BCELoss()

    ########### logging parameters on mlflow
    mlflow.log_params(
        {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "warmup_steps": num_warmup_steps,
            "max_length": config.max_length,
        }
    )

    ########### Training loop ##############

    log_train_loss = []
    log_val_loss = []
    best_loss = 999

    for epoch in range(config.epochs):
        train_losses = 0
        valid_losses = 0
        idx = 0
        model.train()
        for batch in tqdm(train_dataloader):
            batch["input"] = {k: v.to(device) for k, v in batch["input"].items()}
            batch["labels"] = batch["labels"].to(device)

            loss = model.training_step(batch["input"], batch["labels"], loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses += loss.detach().cpu().item()

            ### logging learning rate
            mlflow.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=idx)
            idx += 1

        log_train_loss.append(train_losses / len(train_dataloader))

        model.eval()
        for batch in tqdm(valid_dataloader):
            batch["input"] = {k: v.to(device) for k, v in batch["input"].items()}
            batch["labels"] = batch["labels"].to(device)
            loss = model.training_step(batch["input"], batch["labels"], loss_fn)

            valid_losses += loss.detach().cpu().item()

        log_val_loss.append(valid_losses / len(valid_dataloader))

        if log_val_loss[-1] < best_loss:
            best_loss = log_val_loss[-1]
            torch.save(model, f"../model/{model_name}")

        # Display the losses for each epoch (since epochs are less we'll print for every epoch)
        print(
            f"Epochs :{epoch}  ->  Train loss : {log_train_loss[-1]}  Valid loss : {log_val_loss[-1]}"
        )

        ### logging mlflow
        mlflow.log_metrics(
            {
                "train_loss": log_train_loss[-1],
                "valid_loss": log_val_loss[-1],
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

    history = {"train": log_train_loss, "valid": log_val_loss}

    return model_name, history


def evaluate_model(model):
    ###### Evaluating the trained model ##################################

    tokenizer = DistilBertTokenizer.from_pretrained(
        config.bert_model_path, do_lower=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Moving model to cuda
    model.to(device)
    print(f"Model moved to {device}")

    ## Preparing dataloaders
    eval_dataset = ToxicityDataset(config.test_data_path, tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size)

    ### Eval loop
    f1_avg = 0
    accuracy_avg = 0

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        batch["input"] = {k: v.to(device) for k, v in batch["input"].items()}
        batch["labels"] = batch["labels"].to(device)

        out = model(**batch["input"])

        accuracy, f1 = evaluate(
            out.detach().cpu(), batch["labels"].detach().cpu(), config.threshold
        )
        accuracy_avg += accuracy
        f1_avg += f1

    mlflow.log_metrics(
        {
            "test_f1": f1_avg / len(eval_dataloader),
            "test_accuracy": accuracy_avg / len(eval_dataloader),
        }
    )


if __name__ == "__main__":
    ## Setting up Mlflow pre-reqs
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DistillBert-Toxicity")

    ## Starting mlflow run
    with mlflow.start_run():
        model_name, history = train()
        model = torch.load(f"../model/{model_name}")
        evaluate_model(model)
        mlflow.log_artifact(f"../model/{model_name}", "model")
