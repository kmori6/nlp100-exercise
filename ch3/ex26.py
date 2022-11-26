from typing import Dict
import re
from ex20 import load_json
from ex25 import extract_template


def remove_markup(template: Dict[str, str]) -> Dict[str, str]:
    template = {k: re.sub(r"\'{2,5}", "", v) for k, v in template.items()}
    return template


def main():
    data = load_json()
    template = extract_template(data)
    template = remove_markup(template)
    for k, v in template.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
