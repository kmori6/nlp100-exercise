import numpy as np
import pickle
from ex51 import load_dataset


def calculate_acc(preds: np.ndarray, targets: np.ndarray):
    return (preds == targets).sum() / targets.shape[0]


def main():
    x_test = np.loadtxt("data/test.feature.txt")
    y_test = load_dataset("data/test.txt")["category"].to_numpy()
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(x_test)
    preds = np.argmax(probs, axis=-1)
    acc = calculate_acc(preds, y_test)
    print(acc)


if __name__ == "__main__":
    main()
