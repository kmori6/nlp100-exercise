import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    dataset = pd.read_csv(
        "data/newsCorpora.csv",
        sep="\t",
        names=[
            "id",
            "title",
            "url",
            "publisher",
            "category",
            "story",
            "hostname",
            "timestamp",
        ],
    )
    dataset = dataset[
        dataset["publisher"].isin(
            [
                "Reuters",
                "Huffington Post",
                "Businessweek",
                "Contactmusic.com",
                "Daily Mail",
            ]
        )
    ]
    dataset = dataset.loc[:, ["category", "title"]]
    train_dataset, dev_test_dataset = train_test_split(
        dataset,
        test_size=0.2,
        shuffle=True,
        random_state=0,
        stratify=dataset["category"],
    )
    dev_dataset, test_dataset = train_test_split(
        dev_test_dataset,
        test_size=0.5,
        shuffle=True,
        random_state=0,
        stratify=dev_test_dataset["category"],
    )
    train_dataset.to_csv("data/train.txt", sep="\t", index=False)
    dev_dataset.to_csv("data/valid.txt", sep="\t", index=False)
    test_dataset.to_csv("data/test.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
