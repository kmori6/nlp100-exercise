import os
from urllib.request import urlretrieve
import gzip
import shutil
import json
import re


# ex20
with open("data/jawiki-country.json", "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        if data["title"] == "イギリス":
            break

# ex21
print("ex21")
text = data["text"]
matches = re.findall(r"\[\[Category:.*\]\]", text)
for match in matches:
    print(match)

# ex22
print("ex22")
matches = re.findall(r"\[\[Category:(.*)\]\]", text)
for match in matches:
    print(match)

# ex23
print("ex23")
matches = re.findall(r"(={2,})\s*(.+?)\s*(={2,})", text)
matches = {match[1]: len(match[0]) - 1 for match in matches}
matches = dict(sorted(matches.items(), key=lambda x: x[1]))
for k, v in matches.items():
    print(f"{k}: {v}")

# ex24
print("ex24")
matches = re.findall(r"(\[\[ファイル:)(.+?)(\|)", text)
for match in matches:
    print(match[1])

# ex25
print("ex25")
matches = re.findall(r"(\{\{)(基礎情報\s国\n)(.+?)^(\}\})", text, re.MULTILINE + re.DOTALL)
contents = re.findall(r"(\|)(.+?)(=)(.+?\n)", matches[0][2], re.MULTILINE + re.DOTALL)
template = {}
for content in contents:
    template[content[1].lstrip().rstrip()] = content[-1].lstrip().rstrip()
for k, v in template.items():
    print(f"{k}: {v}")

# ex26
print("ex26")
template = {k: re.sub(r"\'{2,5}", "", v) for k, v in template.items()}
for k, v in template.items():
    print(f"{k}: {v}")

# ex27
print("ex27")
template = {k: re.sub(r"(\[\[)|(\]\])|(.*\|)", "", v) for k, v in template.items()}
for k, v in template.items():
    print(f"{k}: {v}")
