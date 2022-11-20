import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from ex35 import get_word_stats

matplotlib.use("Agg")


def main():
    word_stats = get_word_stats()
    y = list(word_stats.values())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(y, bins=100)
    ax.set_xlabel("出現頻度")
    ax.set_ylabel("単語の種類")
    plt.savefig("data/ex38.png")
    plt.close()


if __name__ == "__main__":
    main()
