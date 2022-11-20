import torch
import torch.nn as nn
from ex70 import NewsDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 300, output_dim: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(feats)
        preds = logits.argmax(-1)
        acc = (preds == targets).sum().item() / len(logits)
        loss = self.cross_entropy_loss(logits, targets)
        return {"logits": logits, "loss": loss, "acc": acc}

    def cross_entropy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        one_hots = nn.functional.one_hot(targets, num_classes=self.output_dim)
        loss = -(one_hots * log_probs).sum(-1)
        if loss.size(0) > 1:
            loss = loss.mean()
        return loss

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(feats)
        probs = torch.softmax(logits, dim=-1)
        return probs


def main():
    train_dataset = NewsDataset("train")
    x1 = train_dataset[0][0]
    X = torch.stack([train_dataset[i][0] for i in range(4)])
    model = NeuralNetwork()
    model.eval()
    with torch.no_grad():
        x_probs = model.predict(x1)
        X_probs = model.predict(X)
    print(x_probs)
    print(X_probs)


if __name__ == "__main__":
    main()
