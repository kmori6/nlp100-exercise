import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ex70 import NewsDataset
from ex71 import NeuralNetwork as BaseNeuralNetwork
from ex73 import train_epoch, validate_epoch
from ex75 import aggregate_log, plot_log
from ex76 import save_checkpoint

matplotlib.use("Agg")


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self, input_dim: int = 300, output_dim: int = 4, hidden_dim: int = 768
    ):
        super().__init__(input_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.bias, 0, 0.01)


def train(train_dataset: Dataset, dev_dataset: Dataset, batch_size: int, epochs: int):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    log = {"train": {"loss": [], "acc": []}, "dev": {"loss": [], "acc": []}}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        dev_loss, dev_acc = validate_epoch(model, dev_dataloader)
        print(
            f"epoch: {epoch}/{epochs}, "
            f"train_loss: {train_loss:.3f}, dev_loss: {dev_loss:.3f}, "
            f"train_acc: {train_acc:.3f}, dev_acc: {dev_acc:.3f}"
        )
        aggregate_log(log, train_loss, train_acc, dev_loss, dev_acc)
        plot_log(epoch, log)
        save_checkpoint(epoch, model, optimizer, dev_loss)

    torch.save(model.state_dict(), "data/model_ex73.pt")


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    train(train_dataset, dev_dataset, batch_size=256, epochs=20)


if __name__ == "__main__":
    main()
