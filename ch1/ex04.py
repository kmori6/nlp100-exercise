text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = text.split()
keys = [
    word[0] if i + 1 in [1, 5, 6, 7, 8, 9, 15, 16, 19] else word[:2]
    for i, word in enumerate(words)
]
dic = {k: i + 1 for i, k in enumerate(keys)}
print(dic)
