import os
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gdown
from gensim.models import KeyedVectors
from ex80 import NewsDataset


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_id: int,
        embed_dim: int = 300,
        hidden_dim: int = 50,
        output_dim: int = 4,
        layers: int = 1,
        bidirectional: bool = False,
        load_embedding_weight: bool = False,
        token_list: List[str] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.padding_id = padding_id
        self.bidirectional = bidirectional
        if load_embedding_weight:
            self.initialize_embedding(vocab_size, embed_dim, padding_id, token_list)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_id)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=layers,
            bidirectional=bidirectional,
            batch_first=True,
            nonlinearity="relu",
        )
        self.classifier = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
        hn = self.rnn(hs)[1]
        if self.bidirectional:
            hs = torch.cat([hn[-2, :], hn[-1, :]], dim=-1)
        else:
            hs = hn.squeeze(0)
        logits = self.classifier(hs)
        preds = logits.argmax(-1)
        acc = (preds == targets).sum().item() / len(logits)
        loss = self.loss_fn(logits, targets)
        return {"logits": logits, "loss": loss, "acc": acc}

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        hs = self.embedding(tokens)
        hn = self.rnn(hs)[1]
        if self.bidirectional:
            hs = torch.cat([hn[-2, :], hn[-1, :]], dim=-1)
        else:
            hs = hn.squeeze(0)
        logits = self.classifier(hs)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        return preds

    def initialize_embedding(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int,
        token_list: List[str],
        file_name: str = "GoogleNews-vectors-negative300.bin.gz",
    ):
        model_path = "data/" + file_name
        if not os.path.exists(model_path):
            gdown.download(id="0B7XkCwpI5KDYNlNUTTlSS21pQmM", output=file_name)
        w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
        weights = torch.randn(vocab_size, embed_dim)
        for i, token in enumerate(token_list):
            if token in w2v.key_to_index:
                weights[i] = torch.tensor(w2v[token])
        self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=padding_idx)


def main():
    train_dataset = NewsDataset("train")
    x1 = train_dataset[0][0]
    X = pad_sequence(
        [train_dataset[i][0] for i in range(4)],
        batch_first=True,
        padding_value=train_dataset.padding_id,
    )
    model = RNNModel(
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
