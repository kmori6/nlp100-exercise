from tqdm import tqdm
from ex60 import load_w2v


def main():
    w2v = load_w2v()
    with open("data/questions-words.txt", "r") as f:
        lines = f.readlines()
    with open("data/results_ex64.txt", "w") as f:
        for line in tqdm(lines):
            if line.startswith(":"):
                f.write(line)
            else:
                words = line.rstrip().split()
                result = w2v.most_similar(
                    positive=[words[1], words[2]], negative=[words[0]]
                )
                word, similarity = result[0]
                f.write(f"{' '.join(words)} {word} {similarity}\n")


if __name__ == "__main__":
    main()
