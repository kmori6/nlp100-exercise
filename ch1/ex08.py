def cipher(text: str):
    result = [str(219 - ord(c)) if c.lower() else c for c in text]
    return result


text = "I am an NLPer"
result = cipher(text)
print(" ".join(result))
