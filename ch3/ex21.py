from typing import Dict, Any
import re
from ex20 import load_json


def extract_rows(data: Dict[str, Any]):
    text = data["text"]
    matches = re.findall(r"\[\[Category:.*\]\]", text)
    return matches


def main():
    data = load_json()
    matches = extract_rows(data)
    for match in matches:
        print(match)


if __name__ == "__main__":
    main()
