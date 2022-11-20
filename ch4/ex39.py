import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from ex35 import get_word_stats

matplotlib.use("Agg")


def main():
    word_stats = get_word_stats()
    x = list(range(len(word_stats)))
    y = list(word_stats.values())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("出現頻度")
    ax.set_ylabel("出現頻度順位")
    plt.savefig("data/ex39.png")
    plt.close()


if __name__ == "__main__":
    main()
