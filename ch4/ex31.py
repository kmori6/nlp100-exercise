from itertools import chain
from ex30 import get_sentences


def main():
    sentences = get_sentences()
    verb_surfaces = [
        [morpheme["surface"] for morpheme in morphemes if morpheme["pos"] == "動詞"]
        for morphemes in sentences
    ]
    verb_surfaces = list(chain(*verb_surfaces))
    verb_surfaces = list(dict.fromkeys(verb_surfaces))
    print(verb_surfaces[:3])


if __name__ == "__main__":
    main()
