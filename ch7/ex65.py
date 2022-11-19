from typing import Dict
import numpy as np


def calculate_acc(results: Dict[str, str]):
    targets = np.array([data["target"] for data in results])
    preds = np.array([data["prediction"] for data in results])
    acc = (preds == targets).sum() / len(targets)
    return acc


def main():
    with open("data/results_ex64.txt", "r") as f:
        lines = f.readlines()
    semantic_results, syntactic_results = [], []
    for line in lines:
        if line.startswith(":"):
            analogy = line.split()[-1]
        else:
            target, prediction, similarity = line.split()[3:]
            results = {
                "target": target,
                "prediction": prediction,
                "similarity": similarity,
            }
            if "gram" in analogy:
                syntactic_results.append(results)
            else:
                semantic_results.append(results)
    acc = {
        "semantic_analogy": calculate_acc(semantic_results),
        "syntactic_analogy": calculate_acc(syntactic_results),
    }
    for k, v in acc.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
