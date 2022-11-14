def n_gram(text: str, n: int):
    words = text.split()
    word_ngram = [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]
    char_ngram = [text[i : i + 2] for i in range(len(text) - 1)]
    return word_ngram, char_ngram


text1 = "paraparaparadise"
text2 = "paragraph"
X = n_gram(text1, 2)[1]
Y = n_gram(text2, 2)[1]
union = set(X + Y)
intersection = set(X).intersection(set(Y))
difference = set(X).difference(set(Y))
print(X)
print(Y)
print(union)
print(intersection)
print(difference)
print("se" in X)
print("se" in Y)
