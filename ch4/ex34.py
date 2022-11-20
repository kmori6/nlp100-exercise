from ex30 import get_sentences


def main():
    sentences = get_sentences()
    nouns = []
    for morphemes in sentences:
        noun = []
        cont = False
        for morpheme in morphemes:
            if morpheme["pos"] == "名詞":
                noun.append(morpheme["surface"])
                cont = True
            else:
                if cont and len(noun) > 1:
                    nouns.append("".join(noun))
                cont = False
                noun = []
    print(nouns[:3])


if __name__ == "__main__":
    main()
