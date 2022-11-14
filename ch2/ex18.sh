#/bin/bash

head -n 3 data/popular-names.txt | sort -k 3 -t " " -r
