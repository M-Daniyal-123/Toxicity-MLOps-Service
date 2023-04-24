from transformers import BertTokenizer, BertModel, BertConfig
import config
from model import ToxicityModel
from dataset import ToxicityDataset
import torch

##Initializing the tokenizer
tokenizer = BertTokenizer.from_pretrained(config.bert_model_path, do_lower=True)


## Initialzing the the model
bert_config = BertConfig.from_pretrained(config.bert_model_path)
bert_model = BertModel(config=bert_config)
model = ToxicityModel(bert_model=bert_model)


## Intializing the dataset class
sample_dataset = ToxicityDataset(
    tokenizer=tokenizer, data_path=config.test_data_path, max_length=config.max_length
)


## testing the dataset class
out = sample_dataset[0]


### Testing the shapes of input
assert out["input"]["input_ids"].shape == torch.Size(
    [config.max_length]
), "Incorrect Max length generated from Dataloader"
### Testing the shape of output
assert out["labels"].shape == torch.Size(
    [config.num_labels]
), "Incorrect Number of labels generated from the Dataloader"

print(f"Dataset class is perfect")


## Testing the output of the model
text = tokenizer(
    "Hello! How are you!",
    padding="max_length",
    max_length=config.max_length,
    return_tensors="pt",
)

out = model(**text)

assert out.shape == torch.Size([1, config.num_labels]), "Incorrect Size from the model"

print(f"The model is perfect")
