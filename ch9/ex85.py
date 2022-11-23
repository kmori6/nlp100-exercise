from ex80 import NewsDataset
from ex81 import RNNModel
from ex82 import train


def main():
    train_dataset = NewsDataset("train")
    dev_dataset = NewsDataset("valid")
    model = RNNModel(
        vocab_size=train_dataset.vocab_size,
        padding_id=train_dataset.padding_id,
        load_embedding_weight=True,
        token_list=list(train_dataset.tokenizer.keys()),
        layers=2,
        bidirectional=True,
    )
    train(train_dataset, dev_dataset, model, batch_size=512)


if __name__ == "__main__":
    main()
