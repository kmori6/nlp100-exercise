#/bin/bash

mkdir -p data

# analogy data
if [ ! -f data/questions-words.txt ]; then
    wget -P data http://download.tensorflow.org/data/questions-words.txt
fi

# word similarity data
if [ ! -f data/combined.csv ]; then
    wget -P data http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
    unzip -d data data/wordsim353.zip
fi
