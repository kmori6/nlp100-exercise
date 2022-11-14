#/bin/bash

l=$1 # 1000 -> 3 files

split -l ${l} data/popular-names.txt data/popular-names.
