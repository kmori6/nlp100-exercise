#/bin/bash

mkdir -p data
url="https://nlp100.github.io/data/jawiki-country.json.gz"
file_name=$(echo ${url} | cut -d "/" -f 5)
if [ ! -f data/${file_name} ]; then
    wget -P data ${url}
fi

gunzip -f data/${file_name}
