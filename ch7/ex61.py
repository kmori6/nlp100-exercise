from ex60 import load_w2v


def main():
    w2v = load_w2v()
    similarity = w2v.similarity("United_States", "U.S.")
    print(similarity)


if __name__ == "__main__":
    main()
