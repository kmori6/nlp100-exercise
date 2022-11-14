#/bin/bash

mkdir -p data
url="https://nlp100.github.io/data/popular-names.txt"
file_name=$(echo ${url} | cut -d "/" -f 5)
if [ ! -f data/${file_name} ]; then
    wget -P data https://nlp100.github.io/data/popular-names.txt
fi
echo $(wc -l data/${file_name})
