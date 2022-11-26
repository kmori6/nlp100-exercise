import json


def load_json():
    with open("data/jawiki-country.json", "r") as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data["title"] == "イギリス":
            break
    return data


def main():
    data = load_json()
    print(data)


if __name__ == "__main__":
    main()
