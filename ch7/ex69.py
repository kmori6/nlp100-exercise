import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from ex60 import load_w2v
from ex67 import get_countries


def main():
    w2v = load_w2v()
    countries = get_countries()
    vecs = [w2v[country] for country in countries]
    tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto", random_state=0)
    embeds = tsne.fit_transform(np.array(vecs))

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(embeds[:, 0], embeds[:, 1])
    plt.savefig("data/ex69.png")


if __name__ == "__main__":
    main()
