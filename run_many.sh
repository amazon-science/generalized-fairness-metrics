
dirr=$1
dirr=${dirr%/}

datasets=( "rt" "imdb" "yelp" "semeval-2" "semeval-3" )

#!/bin/bash
x=1
while [ $x -le 2 ]
do
    for fname in "$dirr"/*
    do
        for i in "${datasets[@]}"
        do
            ./run_experiment.sh exp=${fname} train=1 DATASET=$i
        done
    done
    x=$(( $x + 1 ))
done
