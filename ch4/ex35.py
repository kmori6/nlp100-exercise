from itertools import chain
from collections import Counter
from ex30 import get_sentences


def get_symbols():
    sentences = get_sentences()
    symbols = [
        [morpheme["base"] for morpheme in morphemes if morpheme["pos"] == "記号"]
        for morphemes in sentences
    ]
    symbols = list(chain(*symbols))
    symbols = list(set(symbols))
    return symbols


def get_word_stats():
    sentences = get_sentences()
    raw_words = [
        [morpheme["surface"] for morpheme in morphemes] for morphemes in sentences
    ]
    raw_words = list(chain(*raw_words))
    word_stats = dict(Counter(raw_words))
    symbols = get_symbols()
    for key in symbols:
        del word_stats[key]
    word_stats = dict(sorted(word_stats.items(), key=lambda x: x[1], reverse=True))
    return word_stats


def main():
    word_stats = get_word_stats()
    for i, k in enumerate(word_stats):
        if i > 2:
            break
        print(f"{k}: {word_stats[k]}")


if __name__ == "__main__":
    main()
