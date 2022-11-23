import string
import json
from itertools import chain
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, split: str, unk_id: int = 0):
        super().__init__()
        self.dataset = pd.read_csv(f"data/{split}.txt", sep="\t")
        self.category2label = {"b": 0, "e": 1, "t": 2, "m": 3}
        self.unk_id = unk_id
        self.dataset["category"] = self.dataset["category"].replace(
            self.category2label.keys(), self.category2label.values()
        )
        self.preprocess_text()
        self.tokenizer = self.build_tokenizer(split)
        self.tokenize_text()
        self.padding_id = len(self.tokenizer)
        self.vocab_size = len(self.tokenizer) + 1

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x = torch.tensor(self.dataset["title"][index], dtype=torch.long)
        y = torch.tensor(self.dataset["category"][index], dtype=torch.long)
        return x, y

    def build_tokenizer(self, split: str):
        if split == "train":
            raw_words = [
                [word for word in sentence.split()]
                for sentence in self.dataset["title"]
            ]
            raw_words = list(chain(*raw_words))
            word_stats = dict(Counter(raw_words))
            word_stats = dict(
                sorted(word_stats.items(), key=lambda x: x[1], reverse=True)
            )
            word_stats = {k: v for k, v in word_stats.items() if v >= 2}
            dictionary = {k: i for i, k in enumerate(word_stats.keys(), 1)}
            with open("data/tokenizer.json", "w") as f:
                json.dump(dictionary, f)
        else:
            with open("data/tokenizer.json", "r") as f:
                word_stats = json.load(f)
        return word_stats

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

    def tokenize_text(self):
        for i, data in self.dataset.iterrows():
            words = data["title"].split()
            tokens = [
                self.tokenizer[word] if word in self.tokenizer.keys() else self.unk_id
                for word in words
            ]
            self.dataset.at[i, "title"] = tokens


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    test_dataset = NewsDataset("test")
    print(f"# train samples: {len(train_dataset)}")
    print(f"# dev samples: {len(dev_dataset)}")
    print(f"# test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
