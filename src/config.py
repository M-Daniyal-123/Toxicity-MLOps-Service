columnns = [
    "target",
    "comment_text",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
]
labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
]

train_size = 0.70
test_size = 0.15
validate_size = 0.15
bert_model_path = "distilbert-base-uncased"
max_length = 128
train_data_path = "../data/train.csv"
validation_data_path = "../data/validate.csv"
test_data_path = "../data/test.csv"
num_labels = 6
batch_size = 32
epochs = 1
lr = 0.001
threshold = 0.5
seed = 1337
