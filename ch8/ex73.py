import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex70 import NewsDataset
from ex71 import NeuralNetwork


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
):
    model.train()
    epoch_loss = epoch_crr = 0
    for step, (feats, targets) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        stats = model(feats.to(device), targets.to(device))
        loss = stats["loss"]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feats.size(0)
        epoch_crr += stats["acc"] * feats.size(0)
    epoch_loss /= len(dataloader.dataset)
    epoch_acc = epoch_crr / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, device: torch.device = torch.device("cpu")
):
    model.eval()
    epoch_loss = epoch_crr = 0
    for step, (feats, targets) in enumerate(dataloader, 1):
        with torch.no_grad():
            stats = model(feats.to(device), targets.to(device))
        loss = stats["loss"]
        epoch_loss += loss.item() * feats.size(0)
        epoch_crr += stats["acc"] * feats.size(0)
    epoch_loss /= len(dataloader.dataset)
    epoch_acc = epoch_crr / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for step, (feats, targets) in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            loss = model(feats, targets)["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        dev_loss = 0
        for step, (feats, targets) in enumerate(dev_dataloader, 1):
            with torch.no_grad():
                loss = model(feats, targets)["loss"]
            dev_loss += loss.item()
        dev_loss /= len(dev_dataloader)
        print(
            f"epoch: {epoch}/{epochs}, "
            f"train_loss: {train_loss:.3f}, dev_loss: {dev_loss:.3f}"
        )
    torch.save(model.state_dict(), "data/model_ex73.pt")


if __name__ == "__main__":
    main()
