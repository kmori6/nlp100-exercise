import numpy as np
import pandas as pd
import pickle
from ex51 import category2label


def main():
    with open("data/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    feats_name = vectorizer.get_feature_names_out()
    for k, v in category2label.items():
        print(f"class: {k}")
        rank = np.argsort(model.coef_[v])
        keys = feats_name[rank]
        data = pd.DataFrame([keys[::-1][:10], keys[:10]], index=["top10", "worst10"])
        print(data)


if __name__ == "__main__":
    main()
