import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from ex51 import load_dataset


def main():

    x_train = np.loadtxt("data/train.feature.txt")
    y_train = load_dataset("data/train.txt")["category"].to_numpy()

    model = LogisticRegression(random_state=0, max_iter=500)
    model.fit(x_train, y_train)
    with open("data/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
