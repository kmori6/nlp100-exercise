#/bin/bash

mkdir -p data

if [ ! -f data/newsCorpora.csv ]; then
    wget -P data https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    unzip -d data data/NewsAggregatorDataset.zip
fi

cd ../ch6
python ex50.py
cd ../ch8
cp ../ch6/data/{train,valid,test}.txt data
