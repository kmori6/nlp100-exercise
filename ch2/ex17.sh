#/bin/bash

cat data/popular-names.txt | cut -d " " -f 1 | sort | uniq
