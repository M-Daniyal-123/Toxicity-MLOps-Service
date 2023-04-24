from transformers import DistilBertModel
import torch
import torch.nn as nn

# import config


class ToxicityModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()

        self.bert_model = DistilBertModel.from_pretrained(bert_model)
        self.l1 = nn.Linear(768, 256)  ## Reducing the Vector Dimension
        self.dropout = nn.Dropout(0.2)

        ## ['target','severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.toxicity = nn.Linear(256, 6)  ## 6 classes

        self.bert_model.train()  ## Setting up DistillBert in training mode by default

    def forward(self, **kwargs):
        hc = self.bert_model(**kwargs)
        x = torch.nn.functional.normalize(
            hc[0][:, 0], p=2.0
        )  ## Normalize the embeddings
        x = self.dropout(self.l1(x))
        x = self.toxicity(x)

        return torch.sigmoid(x)

    def training_step(self, input, label, loss_fn):
        out = self(**input)
        loss = loss_fn(out, label)

        return loss
