from typing import Dict, Any, List
import re
from ex20 import load_json


def extract_files(data: Dict[str, Any]) -> List[str]:
    text = data["text"]
    matches = re.findall(r"\[\[ファイル:(.*?)\|", text)
    return matches


def main():
    data = load_json()
    matches = extract_files(data)
    for match in matches:
        print(match)


if __name__ == "__main__":
    main()
