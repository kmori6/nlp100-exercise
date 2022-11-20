import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from ex35 import get_word_stats

matplotlib.use("Agg")


def main():
    word_stats = get_word_stats()
    top10_words = {k: word_stats[k] for i, k in enumerate(word_stats) if i < 10}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(top10_words.keys(), top10_words.values())
    plt.savefig("data/ex36.png")
    plt.close()


if __name__ == "__main__":
    main()
