from itertools import chain
from ex30 import get_sentences


def main():
    sentences = get_sentences()
    verb_bases = [
        [morpheme["base"] for morpheme in morphemes if morpheme["pos"] == "動詞"]
        for morphemes in sentences
    ]
    verb_bases = list(chain(*verb_bases))
    verb_bases = list(dict.fromkeys(verb_bases))
    print(verb_bases[:3])


if __name__ == "__main__":
    main()
