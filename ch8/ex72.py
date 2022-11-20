import torch
from ex70 import NewsDataset
from ex71 import NeuralNetwork


def main():
    train_dataset = NewsDataset("train")
    x1, y1 = train_dataset[0]
    X = torch.stack([train_dataset[i][0] for i in range(4)])
    Y = torch.stack([train_dataset[i][1] for i in range(4)])
    model = NeuralNetwork()
    model.train()

    y1_loss = model(x1.unsqueeze(0), y1.unsqueeze(0))["loss"]
    model.zero_grad()
    y1_loss.backward()
    print(y1_loss)
    print(model.classifier.weight.grad)

    Y_loss = model(X, Y)["loss"]
    model.zero_grad()
    Y_loss.backward()
    print(Y_loss)
    print(model.classifier.weight.grad)


if __name__ == "__main__":
    main()
