import os
import string
import pandas as pd
from tqdm import tqdm
import numpy as np
import gdown
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, split: str):
        super().__init__()
        self.dataset = pd.read_csv(f"data/{split}.txt", sep="\t")
        self.category2label = {"b": 0, "e": 1, "t": 2, "m": 3}
        self.dataset["category"] = self.dataset["category"].replace(
            self.category2label.keys(), self.category2label.values()
        )
        if os.path.exists(f"data/{split}.feats.npy"):
            self.feats = np.load(f"data/{split}.feats.npy")
        else:
            self.preprocess_text()
            self.feats = self.vectorize_text()
            np.save(f"data/{split}.feats.npy", self.feats)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x = torch.tensor(self.feats[index], dtype=torch.float32)
        y = torch.tensor(self.dataset["category"][index], dtype=torch.long)
        return x, y

    def preprocess_text(self):
        self.dataset["title"] = self.dataset["title"].str.lower()
        self.dataset["title"] = self.dataset["title"].str.translate(
            str.maketrans("", "", string.punctuation)
        )
        self.dataset["title"] = self.dataset["title"].replace(
            r"[^0-9a-z\s]", "", regex=True
        )
        self.dataset["title"] = self.dataset["title"].replace(r"\s{2,}", "", regex=True)
        self.dataset["title"] = self.dataset["title"].replace(r"\s*$", "", regex=True)

    def vectorize_text(
        self, file_name: str = "GoogleNews-vectors-negative300.bin.gz"
    ) -> np.ndarray:
        model_path = "data/" + file_name
        if not os.path.exists(model_path):
            gdown.download(id="0B7XkCwpI5KDYNlNUTTlSS21pQmM", output=file_name)
        w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
        feats = []
        for text in tqdm(self.dataset["title"]):
            words = text.split()
            vecs = np.array([w2v[word] for word in words if word in w2v]).mean(0)
            feats.append(vecs)
        feats = np.stack(feats)
        return feats


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    test_dataset = NewsDataset("test")
    print(f"# train samples: {len(train_dataset)}")
    print(f"# dev samples: {len(dev_dataset)}")
    print(f"# test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
