import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex70 import NewsDataset
from ex71 import NeuralNetwork

matplotlib.use("Agg")


def aggregate_log(log, train_loss, train_acc, dev_loss, dev_acc):
    log["train"]["loss"].append(train_loss)
    log["train"]["acc"].append(train_acc)
    log["dev"]["loss"].append(dev_loss)
    log["dev"]["acc"].append(dev_acc)


def plot_log(epoch: int, log):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(range(1, epoch + 1)), log["train"]["loss"], label="train")
    ax.plot(list(range(1, epoch + 1)), log["dev"]["loss"], label="dev")
    plt.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("loss plots")
    plt.savefig("data/loss_ex75.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(range(1, epoch + 1)), log["train"]["acc"], label="train")
    ax.plot(list(range(1, epoch + 1)), log["dev"]["acc"], label="dev")
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc")
    ax.set_title("acc plots")
    plt.legend()
    plt.savefig("data/acc_ex75.png")
    plt.close()


def train(
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
):
    log = {"train": {"loss": [], "acc": []}, "dev": {"loss": [], "acc": []}}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_acc = 0
        for step, (feats, targets) in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            stats = model(feats, targets)
            loss = stats["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += stats["acc"]
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        log["train"]["loss"].append(train_loss)
        log["train"]["acc"].append(train_acc)

        model.eval()
        dev_loss = dev_acc = 0
        for step, (feats, targets) in enumerate(dev_dataloader, 1):
            with torch.no_grad():
                stats = model(feats, targets)
            loss = stats["loss"]
            dev_loss += loss.item()
            dev_acc += stats["acc"]
        dev_loss /= len(dev_dataloader)
        dev_acc /= len(dev_dataloader)
        log["dev"]["loss"].append(dev_loss)
        log["dev"]["acc"].append(dev_acc)

        print(
            f"epoch: {epoch}/{epochs}, "
            f"train_loss: {train_loss:.3f}, dev_loss: {dev_loss:.3f}, "
            f"train_acc: {train_acc:.3f}, dev_acc: {dev_acc:.3f}"
        )

        plot_log(epoch, log)

    torch.save(model.state_dict(), "data/model_ex73.pt")


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    epochs = 10
    train(train_dataloader, dev_dataloader, model, optimizer, epochs)


if __name__ == "__main__":
    main()
