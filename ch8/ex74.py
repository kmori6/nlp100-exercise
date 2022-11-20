import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex70 import NewsDataset
from ex71 import NeuralNetwork


def calculate_acc(preds: torch.Tensor, targets: torch.Tensor):
    return (preds == targets).sum().item() / preds.size(0)


def load_model(model: nn.Module, model_path: str, device: torch.device):
    model.load_state_dict(torch.load(model_path, map_location=device))


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    train_dataloader = DataLoader(train_dataset, batch_size=256)
    dev_dataloader = DataLoader(dev_dataset, batch_size=256)
    model = NeuralNetwork()
    load_model(model, model_path="data/model_ex73.pt", device=torch.device("cpu"))
    model.eval()
    acc = {}
    all_preds, all_targets = [], []
    for feats, targets in train_dataloader:
        with torch.no_grad():
            probs = model.predict(feats)
        preds = probs.argmax(-1)
        all_preds.append(preds)
        all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc["train"] = calculate_acc(all_preds, all_targets)

    all_preds, all_targets = [], []
    for feats, targets in dev_dataloader:
        with torch.no_grad():
            probs = model.predict(feats)
        preds = probs.argmax(-1)
        all_preds.append(preds)
        all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc["dev"] = calculate_acc(all_preds, all_targets)

    print(acc)


if __name__ == "__main__":
    main()
