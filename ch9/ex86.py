import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ex80 import NewsDataset


class CNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_id: int,
        embed_dim: int = 300,
        hidden_dim: int = 50,
        output_dim: int = 4,
    ):
        super().__init__()
        self.padding_id = padding_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_id)
        self.cnn = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
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
        hs = self.cnn(hs.transpose(1, 2)).transpose(1, 2)
        hs = self.activation(hs)
        hs = torch.max(hs, dim=1).values
        logits = self.classifier(hs)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        return preds


def main():
    train_dataset = NewsDataset("train")
    x1 = train_dataset[0][0].unsqueeze(0)
    X = pad_sequence(
        [train_dataset[i][0] for i in range(4)],
        batch_first=True,
        padding_value=train_dataset.padding_id,
    )
    model = CNNModel(
        vocab_size=train_dataset.vocab_size, padding_id=train_dataset.padding_id
    )
    model.eval()
    with torch.no_grad():
        x_preds = model.predict(x1)
        X_preds = model.predict(X)
    print(x_preds)
    print(X_preds)


if __name__ == "__main__":
    main()
