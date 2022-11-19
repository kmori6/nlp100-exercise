#/bin/bash

mkdir -p data

if [ ! -f data/newsCorpora.csv ]; then
    wget -P data https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    unzip -d data data/NewsAggregatorDataset.zip
fi
