#/bin/bash

cat data/${file_name} | cut -d " " -f 1 > data/col1.txt
cat data/${file_name} | cut -d " " -f 2 > data/col2.txt
