from typing import Dict
import re
from ex20 import load_json
from ex25 import extract_template
from ex26 import remove_markup
from ex27 import remove_link


def remove_others(template: Dict[str, str]) -> Dict[str, str]:
    # remove files
    template = {
        k: re.sub(r"\[\[ファイル:([^\|]+?)\|(.+?)\]\]", r"\1", v)
        for k, v in template.items()
    }
    # remove refs
    template = {
        k: re.sub(r"<ref.+?/ref>", "", v.replace("\n", "")) for k, v in template.items()
    }
    template = {k: re.sub(r"<.+?/>", "", v) for k, v in template.items()}
    template = {k: re.sub(r"\{\{([^\|]+?)\}\}", r"\1", v) for k, v in template.items()}
    template = {
        k: re.sub(r"\{\{[^\|]+?\|[^\|]+?\|([^\|]+?)\}\}", r"\1", v)
        for k, v in template.items()
    }
    template = {
        k: re.sub(r"\{\{[^\|]+?\|[^\|]+?\|[^\|]+?\|([^\|]+?)\}\}", r"\1", v)
        for k, v in template.items()
    }
    template = {
        k: re.sub(r"\[\[[^\|]+?\|([^\|]+?)\]\].+?$", r"\1", v)
        for k, v in template.items()
    }
    return template


def main():
    data = load_json()
    template = extract_template(data)
    template = remove_markup(template)
    template = remove_link(template)
    template = remove_others(template)
    for k, v in template.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
