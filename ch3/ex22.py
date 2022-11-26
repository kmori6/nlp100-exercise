from typing import Dict, Any
import re
from ex20 import load_json


def extract_categories(data: Dict[str, Any]):
    text = data["text"]
    matches = re.findall(r"\[\[Category:(.+?)(\|.+?)?\]\]", text)
    return [match[0] for match in matches]


def main():
    data = load_json()
    matches = extract_categories(data)
    for match in matches:
        print(match)


if __name__ == "__main__":
    main()
