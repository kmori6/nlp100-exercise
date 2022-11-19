from typing import List
import pandas as pd
from sklearn.cluster import KMeans
from ex60 import load_w2v


def get_countries():
    with open("data/questions-words.txt", "r") as f:
        lines = f.readlines()
    countries = []
    for line in lines:
        if line.startswith(":"):
            analogy = line.split()[-1]
        else:
            country = None
            if analogy in ["capital-common-countries", "capital-world"]:
                country = line.split()[1]
            elif analogy == "currency":
                country = line.split()[0]
            if country is not None:
                countries.append(country)
    countries = list(set(countries))
    return countries


def kmeans_clustering(countries: List[str]):
    w2v = load_w2v()
    vecs = [w2v[country] for country in countries]
    kmeans = KMeans(n_clusters=5, random_state=0).fit(vecs)
    return kmeans


def main():
    countries = get_countries()
    kmeans = kmeans_clustering(countries)
    df = pd.DataFrame({"country": countries, "cluster": kmeans.labels_})
    print(df)


if __name__ == "__main__":
    main()
