import os
import gdown
from gensim.models import KeyedVectors


def load_w2v(file_name: str = "GoogleNews-vectors-negative300.bin.gz"):
    os.makedirs("data", exist_ok=True)
    model_path = "data/" + file_name
    if not os.path.exists(model_path):
        gdown.download(id="0B7XkCwpI5KDYNlNUTTlSS21pQmM", output=file_name)
    w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return w2v


def main():
    w2v = load_w2v()
    print(w2v["United_States"])


if __name__ == "__main__":
    main()
