from ex30 import get_sentences


def main():
    sentences = get_sentences()
    a_no_b = []
    for morphemes in sentences:
        for i in range(len(morphemes) - 2):
            if (
                morphemes[i]["pos"] == "名詞"
                and morphemes[i + 1]["surface"] == "の"
                and morphemes[i + 2]["pos"] == "名詞"
            ):
                a_no_b.append(
                    morphemes[i]["surface"]
                    + morphemes[i + 1]["surface"]
                    + morphemes[i + 2]["surface"]
                )
    print(a_no_b[:3])


if __name__ == "__main__":
    main()
