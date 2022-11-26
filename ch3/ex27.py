from typing import Dict
import re
from ex20 import load_json
from ex25 import extract_template
from ex26 import remove_markup


def remove_link(template: Dict[str, str]) -> Dict[str, str]:
    # [[記事名]] -> 記事名
    template = {k: re.sub(r"\[\[([^\|]+?)\]\]", r"\1", v) for k, v in template.items()}
    # [[記事名|表示文字]] -> 表示文字
    template = {
        k: re.sub(r"\[\[[^\|]+?\|([^\|]+?)\]\]", r"\1", v) for k, v in template.items()
    }
    # [[記事名#節名|表示文字]] -> 表示文字
    template = {
        k: re.sub(r"\[\[[^\|]+?#.+?\|([^\|]+?)\]\]", r"\1", v)
        for k, v in template.items()
    }
    return template


def main():
    data = load_json()
    template = extract_template(data)
    template = remove_markup(template)
    template = remove_link(template)
    for k, v in template.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
