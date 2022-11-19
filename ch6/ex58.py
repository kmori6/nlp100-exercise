import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from ex51 import load_dataset
from ex54 import calculate_acc


def main():
    x_train = np.loadtxt("data/train.feature.txt")
    x_dev = np.loadtxt("data/valid.feature.txt")
    x_test = np.loadtxt("data/test.feature.txt")
    y_train = load_dataset("data/train.txt")["category"].to_numpy()
    y_dev = load_dataset("data/valid.txt")["category"].to_numpy()
    y_test = load_dataset("data/test.txt")["category"].to_numpy()

    results = []
    for c in tqdm(np.logspace(-1, 1, 10, base=10)):
        model = LogisticRegression(random_state=0, max_iter=500, C=c)
        model.fit(x_train, y_train)
        preds = {
            "train": model.predict(x_train),
            "dev": model.predict(x_dev),
            "test": model.predict(x_test),
        }
        acc = {
            "train": calculate_acc(preds["train"], y_train),
            "dev": calculate_acc(preds["dev"], y_dev),
            "test": calculate_acc(preds["test"], y_test),
        }
        results.append({"c": c, "acc": acc})

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    c = [result["c"] for result in results]
    ax.plot(c, [result["acc"]["train"] for result in results], label="train")
    ax.plot(c, [result["acc"]["dev"] for result in results], label="dev")
    ax.plot(c, [result["acc"]["test"] for result in results], label="test")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("c")
    ax.set_xscale("log")
    plt.legend()
    plt.savefig("data/ex58.png")


if __name__ == "__main__":
    main()
