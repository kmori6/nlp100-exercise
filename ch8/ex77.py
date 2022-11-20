import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from ex70 import NewsDataset
from ex71 import NeuralNetwork

matplotlib.use("Agg")


def train(train_dataset: Dataset, batch_size: int):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    model.train()
    start_time = time.time()
    for feats, targets in train_dataloader:
        optimizer.zero_grad()
        loss = model(feats, targets)["loss"]
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return end_time - start_time


def main():
    train_dataset = NewsDataset("train")
    results = []
    for batch_size in tqdm(np.logspace(0, 8, 9, base=2, dtype=np.int64)):
        time = train(train_dataset, int(batch_size))
        results.append(time)
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.logspace(0, 8, 9, base=2, dtype=int), results)
    ax.set_xlabel("batch size")
    ax.set_ylabel("sec")
    plt.savefig("data/ex77.png")


if __name__ == "__main__":
    main()
