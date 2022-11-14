text1 = "パトカー"
text2 = "タクシー"
text3 = [text1[i] + text2[i] for i in range(len(text1))]
print("".join(text3))
