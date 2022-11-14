import random


def process(text: str):
    words = text.split()
    result = []
    for word in words:
        if len(word) > 4:
            parts = list(word[1:-1])
            random.shuffle(parts)
            word = word[0] + "".join(parts) + word[-1]
        result.append(word)
    return result


text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(process(text))
