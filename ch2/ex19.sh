#/bin/bash

cut -d " " -f 1 data/popular-names.txt | sort | uniq -c | sort -r
