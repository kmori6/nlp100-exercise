from ex60 import load_w2v


def main():
    w2v = load_w2v()
    similar_words = w2v.most_similar(
        positive=["Spain", "Athens"], negative=["Madrid"], topn=10
    )
    print(similar_words)


if __name__ == "__main__":
    main()
