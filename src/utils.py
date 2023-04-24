from sklearn.metrics import f1_score, accuracy_score
import torch


def evaluate(predictions, labels, threshold):
    predictions = torch.where(predictions >= threshold, 1, 0)

    accuracy, f1 = accuracy_score(predictions, labels), f1_score(
        predictions, labels, average="samples", zero_division=1.0
    )

    lb_name = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
    ]

    return accuracy, f1
