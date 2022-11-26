from typing import Dict, Any, List
import re
from ex20 import load_json


def extract_template(data: Dict[str, Any]) -> List[str]:
    text = data["text"]
    template = re.findall(r"\{\{基礎情報\s国\n.+?\n\}\}", text, re.DOTALL)[0]
    fields = re.findall(r"\|.+?\s*=\s*.+\n", template, re.DOTALL)[0]
    contents = []
    for line in fields.split("\n"):
        if "=" in line:
            contents.append(line)
        else:
            contents[-1] += "\n" + line
    results = {}
    for content in contents:
        key, value = content.split("=", 1)
        results[key.replace("|", "").rstrip()] = value.lstrip()
    return results


def main():
    data = load_json()
    template = extract_template(data)
    for k, v in template.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
