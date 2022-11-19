import numpy as np
import pickle


def main():
    x_test = np.loadtxt("data/test.feature.txt")
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(x_test)
    print(probs)


if __name__ == "__main__":
    main()
