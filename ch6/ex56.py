import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from ex51 import load_dataset, category2label


def main():
    x_test = np.loadtxt("data/test.feature.txt")
    y_test = load_dataset("data/test.txt")["category"].to_numpy()
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(x_test)
    preds = np.argmax(probs, axis=-1)

    precision = precision_score(y_test, preds, average=None)
    recall = recall_score(y_test, preds, average=None)
    f1 = f1_score(y_test, preds, average=None)

    results = pd.DataFrame(
        {"適合率": precision, "再現率": recall, "F1スコア": f1}, index=category2label.keys()
    )
    results.loc["micro-average"] = [
        precision_score(y_test, preds, average="micro"),
        recall_score(y_test, preds, average="micro"),
        f1_score(y_test, preds, average="micro"),
    ]
    results.loc["macro-average"] = [
        precision_score(y_test, preds, average="macro"),
        recall_score(y_test, preds, average="macro"),
        f1_score(y_test, preds, average="macro"),
    ]
    print(results)


if __name__ == "__main__":
    main()
