import numpy as np
from sklearn.linear_model import LogisticRegression
from ex51 import load_dataset
from ex54 import calculate_acc
import optuna


def main():
    x_train = np.loadtxt("data/train.feature.txt")
    x_dev = np.loadtxt("data/valid.feature.txt")
    x_test = np.loadtxt("data/test.feature.txt")
    y_train = load_dataset("data/train.txt")["category"].to_numpy()
    y_dev = load_dataset("data/valid.txt")["category"].to_numpy()
    y_test = load_dataset("data/test.txt")["category"].to_numpy()

    def objective(trial):
        c = trial.suggest_float("c", 1e-5, 1e5, log=True)

        model = LogisticRegression(random_state=0, max_iter=500, C=c)
        model.fit(x_train, y_train)

        dev_preds = model.predict(x_dev)
        dev_acc = (dev_preds == y_dev).sum() / y_dev.shape[0]

        return dev_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    trial = study.best_trial
    print(f"best objective: {trial.value}")
    for k, v in trial.params.items():
        print(f"{k}: {v}")

    model = LogisticRegression(random_state=0, max_iter=500, C=trial.params["c"])
    model.fit(x_train, y_train)

    preds = {
        "dev": model.predict(x_dev),
        "test": model.predict(x_test),
    }
    acc = {
        "dev": calculate_acc(preds["dev"], y_dev),
        "test": calculate_acc(preds["test"], y_test),
    }
    print(f"dev_acc: {acc['dev']}")
    print(f"test_acc: {acc['test']}")


if __name__ == "__main__":
    main()
