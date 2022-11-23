import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ex80 import NewsDataset
from ex81 import RNNModel


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


def train(
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: nn.Module,
    batch_size: int,
    epochs: int = 10,
    lr: float = 1e-3,
):
    def collate_fn(batch):
        tokens = pad_sequence(
            [sample[0] for sample in batch],
            batch_first=True,
            padding_value=model.padding_id,
        )
        targets = torch.tensor([sample[1] for sample in batch])
        return tokens, targets

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        dev_loss, dev_acc = validate_epoch(model, dev_dataloader, device)
        print(
            f"epoch: {epoch}/{epochs}, "
            f"train_loss: {train_loss:.3f}, dev_loss: {dev_loss:.3f}, "
            f"train_acc: {train_acc:.3f}, dev_acc: {dev_acc:.3f}"
        )
    torch.save(model.state_dict(), "data/model_ex82.pt")


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    model = RNNModel(
        vocab_size=train_dataset.vocab_size, padding_id=train_dataset.padding_id
    )
    train(train_dataset, dev_dataset, model, batch_size=1)


if __name__ == "__main__":
    main()
