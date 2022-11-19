import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from ex51 import load_dataset


def main():
    x_test = np.loadtxt("data/test.feature.txt")
    y_test = load_dataset("data/test.txt")["category"].to_numpy()
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(x_test)
    preds = np.argmax(probs, axis=-1)
    cm = confusion_matrix(preds, y_test)
    print(cm)


if __name__ == "__main__":
    main()
