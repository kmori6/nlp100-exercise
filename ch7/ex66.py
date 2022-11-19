from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from ex60 import load_w2v


def main():
    w2v = load_w2v()
    df = pd.read_csv("data/combined.csv")
    targets = df["Human (mean)"].tolist()
    preds = [w2v.similarity(data[1], data[2]) for data in df.itertuples()]
    correlation = spearmanr(targets, preds)[0]
    print(correlation)


if __name__ == "__main__":
    main()
