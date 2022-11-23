from ex80 import NewsDataset
from ex82 import train
from ex86 import CNNModel


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    model = CNNModel(
        vocab_size=train_dataset.vocab_size, padding_id=train_dataset.padding_id
    )
    train(train_dataset, dev_dataset, model, batch_size=512)


if __name__ == "__main__":
    main()
