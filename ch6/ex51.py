import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

category2label = {"b": 0, "e": 1, "t": 2, "m": 3}


def load_dataset(path: str):
    dataset = pd.read_csv(path, sep="\t")
    # preprocess
    dataset["title"] = dataset["title"].str.lower()
    dataset["category"] = dataset["category"].replace(
        category2label.keys(), category2label.values()
    )
    return dataset


def main():

    train_dataset = load_dataset("data/train.txt")
    dev_dataset = load_dataset("data/valid.txt")
    test_dataset = load_dataset("data/test.txt")

    vectorizer = TfidfVectorizer(max_features=500)
    train_feats = vectorizer.fit_transform(train_dataset["title"]).toarray()
    dev_feats = vectorizer.transform(dev_dataset["title"]).toarray()
    test_feats = vectorizer.transform(test_dataset["title"]).toarray()

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    np.savetxt("data/train.feature.txt", train_feats)
    np.savetxt("data/valid.feature.txt", dev_feats)
    np.savetxt("data/test.feature.txt", test_feats)


if __name__ == "__main__":
    main()
