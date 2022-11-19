from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from ex60 import load_w2v
from ex67 import get_countries


def main():
    w2v = load_w2v()
    countries = get_countries()
    vecs = [w2v[country] for country in countries]

    sns.set()
    plt.figure()
    Z = linkage(vecs, "ward")
    dendrogram(Z, labels=countries)
    plt.savefig("data/ex68.png")


if __name__ == "__main__":
    main()
