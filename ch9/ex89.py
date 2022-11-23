import string
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from ex82 import train


class NewsDataset(Dataset):
    def __init__(self, split: str):
        super().__init__()
        self.dataset = pd.read_csv(f"data/{split}.txt", sep="\t")
        self.category2label = {"b": 0, "e": 1, "t": 2, "m": 3}
        self.dataset["category"] = self.dataset["category"].replace(
            self.category2label.keys(), self.category2label.values()
        )
        self.preprocess_text()
        self.tokenizer = self.build_tokenizer()
        self.tokenize_text()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x = torch.tensor(self.dataset["title"][index], dtype=torch.long)
        y = torch.tensor(self.dataset["category"][index], dtype=torch.long)
        return x, y

    def build_tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_text(self):
        self.dataset["title"] = self.dataset["title"].str.lower()
        self.dataset["title"] = self.dataset["title"].str.translate(
            str.maketrans("", "", string.punctuation)
        )
        self.dataset["title"] = self.dataset["title"].replace(
            r"[^0-9a-z\s]", "", regex=True
        )
        self.dataset["title"] = self.dataset["title"].replace(r"\s{2,}", "", regex=True)
        self.dataset["title"] = self.dataset["title"].replace(r"\s*$", "", regex=True)

    def tokenize_text(self):
        for i, data in self.dataset.iterrows():
            text = data["title"]
            tokens = self.tokenizer(text)["input_ids"]
            self.dataset.at[i, "title"] = tokens


class BertModel(nn.Module):
    def __init__(self, output_dim: int = 4, padding_id: int = 0):
        super().__init__()
        self.padding_id = padding_id
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.pooler.dense.in_features, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        masks = torch.where(tokens != self.padding_id, 1, 0)
        hs = self.bert(tokens, masks)[0][:, 0, :]
        logits = self.classifier(hs)
        preds = logits.argmax(-1)
        acc = (preds == targets).sum().item() / len(logits)
        loss = self.loss_fn(logits, targets)
        return {"logits": logits, "loss": loss, "acc": acc}

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        masks = torch.where(tokens != self.padding_id, 1, 0)
        hs = self.bert(tokens, masks)[0][:, 0, :]
        logits = self.classifier(hs)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        return preds


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    model = BertModel()
    train(train_dataset, dev_dataset, model, batch_size=512, epochs=20)


if __name__ == "__main__":
    main()
