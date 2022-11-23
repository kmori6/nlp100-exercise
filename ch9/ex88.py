import torch
import torch.nn as nn
from typing import List
from ex80 import NewsDataset
from ex81 import RNNModel
from ex82 import train


class RNNCNNModel(RNNModel):
    def __init__(
        self,
        vocab_size: int,
        padding_id: int,
        embed_dim: int = 300,
        hidden_dim: int = 50,
        output_dim: int = 4,
        rnn_layers: int = 2,
        load_embedding_weight: bool = False,
        token_list: List[str] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            padding_id=padding_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            layers=rnn_layers,
            load_embedding_weight=load_embedding_weight,
            token_list=token_list,
        )
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.cnn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
        hs = self.rnn(hs)[0]
        hs = self.cnn(hs.transpose(1, 2)).transpose(1, 2)
        hs = self.activation(hs)
        hs = torch.max(hs, dim=1).values
        logits = self.classifier(hs)
        preds = logits.argmax(-1)
        acc = (preds == targets).sum().item() / len(logits)
        loss = self.loss_fn(logits, targets)
        return {"logits": logits, "loss": loss, "acc": acc}

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
        hs = self.rnn(hs)[0]
        hs = self.cnn(hs.transpose(1, 2)).transpose(1, 2)
        hs = self.activation(hs)
        hs = torch.max(hs, dim=1).values
        logits = self.classifier(hs)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        return preds


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    model = RNNCNNModel(
        vocab_size=train_dataset.vocab_size,
        padding_id=train_dataset.padding_id,
        load_embedding_weight=True,
        token_list=list(train_dataset.tokenizer.keys()),
    )
    train(train_dataset, dev_dataset, model, batch_size=512, epochs=100)


if __name__ == "__main__":
    main()
