#/bin/bash

mkdir -p data
url="https://nlp100.github.io/data/neko.txt"
file_name=$(echo ${url} | cut -d "/" -f 5)
if [ ! -f data/${file_name} ]; then
    wget -P data ${url}
fi

sed -i "" -e "1,2d; s/　//g" data/neko.txt

mecab -o data/neko.txt.mecab data/neko.txt
