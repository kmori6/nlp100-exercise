from typing import Dict, Any
import re
from ex20 import load_json


def extract_sections(data: Dict[str, Any]) -> Dict[str, int]:
    text = data["text"]
    matches = re.findall(r"(={2,})\s*(.*?)\s*(={2,})", text)
    matches = {match[1]: len(match[0]) - 1 for match in matches}
    matches = dict(sorted(matches.items(), key=lambda x: x[1]))
    return matches


def main():
    data = load_json()
    matches = extract_sections(data)
    for k, v in matches.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
