#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" 
DATA="$ROOT/datasets"

OUTDIR="models"

DATASET="semeval-2"

train=1

source activate btools

# template for parsing arguments from
# https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case $KEY in
    # general arguments
        ROOT)               ROOT=${VALUE} ;;
        OUTDIR)             OUTDIR=${VALUE} ;;  
    # experiment specific arguments
        DATASET)            DATASET=${VALUE} ;; 
        # path to the experiment file 
        exp)                exp=${VALUE} ;; 
        # 0/1 train the model or just evaluate
        train)              train=${VALUE} ;;
        *)   
    esac    
done

# this might be later reset based on the dataset (line 50)
TASK="class"

# set other variables based on the dataset
case $DATASET in 
    semeval-2) VAL_DATA="${DATA}/semeval-v-oc/2018-Valence-oc-En-dev.txt"  &&
            TRAIN_DATA="${DATA}/semeval-v-oc/2018-Valence-oc-En-train.txt"  &&
            READER="SEMEVAL" &&
            GRANULARITY="2" ;;
    semeval-3) VAL_DATA="${DATA}/semeval-v-oc/2018-Valence-oc-En-dev.txt"  &&
        TRAIN_DATA="${DATA}/semeval-v-oc/2018-Valence-oc-En-train.txt"  &&
        READER="SEMEVAL" &&
        GRANULARITY="3" ;;
    conll2003) VAL_DATA="${DATA}/conll2003/ner/eng.testa" &&
        TRAIN_DATA="${DATA}/conll2003/ner/eng.train" &&
        TASK="seq" ;;
    *) TEST_DATA="" && VAL_DATA="" && TRAIN_DATA="";
esac


error=0
[ -z "$TRAIN_DATA" ] &&  error=1 &&  echo "ERROR: unsupported dataset $DATASET"
[ -z "$exp" ] &&  error=1 &&  echo "ERROR: exp argument is required, please provide the jsonnet config file"
#[ -z "$train" ] &&  error=1 && echo "ERROR: train argument is required, please specify whether the model should be trained (1) or not (0)"
[ $error == 1 ] && exit


mkdir -p $OUTDIR


printf "\n>>>>>>> executing bash script with the following arguments:\n"
echo "ROOT: ${ROOT}"
echo "OUTDIR: ${OUTDIR}"
echo "VAL_DATA: ${VAL_DATA}"
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "exp: ${exp}"
echo "train: ${train}"
printf "\n"

cd $ROOT

#################################################
#       THE MAIN SCRIPT STARTS HERE             #
#################################################

# get model name from the name of the json file
mname=$(basename $exp)
mname=${mname%.*}

modeldir="${OUTDIR}/${mname}-${DATASET}"

if [[ "$modeldir" == *[^0-9] ]]; then
    modeldir="${modeldir}-${GRANULARITY}"
fi

x=1
orig="${modeldir}"
while [ -d "$modeldir" ]
do
    modeldir="${orig}-$x"
    x=$(( $x + 1 ))
done

if [ $train == 1 ]; then
    echo "Training the model: $mname"
    echo "Output dir: $modeldir"
    printf "\n"

    if [ "$TASK" = "class" ]; then
        oargs="{'dataset_reader': {\
                    'reader': '${READER}',\
                    'granularity': '${GRANULARITY}-class',\
                },\
                'validation_dataset_reader': {\
                    'reader': '${READER}',\
                    'granularity': '${GRANULARITY}-class',\
                },\
                'train_data_path': '${TRAIN_DATA}',\
                'validation_data_path': '${VAL_DATA}',\
                'model': {\
                    'num_labels': ${GRANULARITY},\
                },\
                }"
    else
        oargs="{'train_data_path': '${TRAIN_DATA}',\
                'validation_data_path': '${VAL_DATA}'\
                }"
    fi

    echo $oargs

   allennlp train "${exp}" -s "${modeldir}" -f -o "${oargs}" --include-package src.models

fi


