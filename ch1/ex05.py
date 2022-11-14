def n_gram(text: str, n: int):
    words = text.split()
    word_ngram = [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]
    char_ngram = [text[i : i + 2] for i in range(len(text) - 1)]
    return word_ngram, char_ngram


text = "I am an NLPer"
word_ngram, char_ngram = n_gram(text, 2)
print(word_ngram)
print(char_ngram)
