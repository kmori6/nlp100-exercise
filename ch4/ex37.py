from itertools import chain
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
from ex30 import get_sentences
from ex35 import get_symbols

matplotlib.use("Agg")


def main():
    sentences = get_sentences()
    symbols = get_symbols()
    cat_sentences = []
    for morphemes in sentences:
        in_cat = False
        for morpheme in morphemes:
            surface = morpheme["surface"]
            if surface == "猫":
                in_cat = True
                break
        if in_cat:
            cat_sentences.append(morphemes)
    raw_cat_words = [
        [morpheme["surface"] for morpheme in morphemes] for morphemes in cat_sentences
    ]
    raw_cat_words = list(chain(*raw_cat_words))
    cat_word_stats = dict(Counter(raw_cat_words))
    for key in symbols:
        if key in cat_word_stats.keys():
            del cat_word_stats[key]
    del cat_word_stats["猫"]
    cat_word_stats = dict(
        sorted(cat_word_stats.items(), key=lambda x: x[1], reverse=True)
    )
    top10_cat_words = {
        k: cat_word_stats[k] for i, k in enumerate(cat_word_stats) if i < 10
    }
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(top10_cat_words.keys(), top10_cat_words.values())
    plt.savefig("data/ex37.png")
    plt.close()


if __name__ == "__main__":
    main()
