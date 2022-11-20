def get_sentences():
    with open("data/neko.txt.mecab", "r") as f:
        lines = f.readlines()
    sentences, morphemes = [], []
    for content in lines:
        if content.rstrip() != "EOS" and len(content) > 1:
            morpheme = content.split(",")
            surface, pos = morpheme[0].split("\t")
            base = morpheme[6]
            pos1 = morpheme[1]
            morpheme = {"surface": surface, "base": base, "pos": pos, "pos1": pos1}
            morphemes.append(morpheme)
        else:
            if len(morphemes) > 1:
                sentences.append(morphemes)
            morphemes = []
    return sentences


def main():
    sentences = get_sentences()
    print(sentences[0])
    print(sentences[-1])


if __name__ == "__main__":
    main()
