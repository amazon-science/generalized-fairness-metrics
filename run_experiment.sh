# example usage:
# ./run_experiment.sh exp=experiments/roberta.jsonnet DATASET=mixed

# all of the following are default arguments
# which can be overriden by the arguments given to this script

ROOT="/home/ubuntu/workplace/ComprehendBiasTools"
DATA="$ROOT/datasets"

OUTDIR="/home/ubuntu/czarpaul/trained_models"

DATASET="sst-2"

train=1


# template for parsing arguments taken from
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

# this is reset based on the dataset
TASK="class"

# set other variables based on the dataset
case $DATASET in 
    sst-2) TEST_DATA="${DATA}/SST/trees/test.txt" &&
            VAL_DATA="${DATA}/SST/trees/dev.txt" &&
            TRAIN_DATA="${DATA}/SST/trees/train.txt" &&
            READER="SST" &&
            GRANULARITY="2" ;;
    sst-3) TEST_DATA="${DATA}/SST/trees/test.txt" &&
            VAL_DATA="${DATA}/SST/trees/dev.txt" &&
            TRAIN_DATA="${DATA}/SST/trees/train.txt" &&
            READER="SST" &&
            GRANULARITY="3" ;;
    sst-5) TEST_DATA="${DATA}/SST/trees/test.txt" &&
            VAL_DATA="${DATA}/SST/trees/dev.txt" &&
            TRAIN_DATA="${DATA}/SST/trees/train.txt" &&
            READER="SST" &&
            GRANULARITY="5" ;;
    rt) TEST_DATA="rotten_tomatoes@test" &&
            VAL_DATA="rotten_tomatoes@dev"  &&
            TRAIN_DATA="rotten_tomatoes@train"  &&
            READER="HUGGINGFACE" &&
            GRANULARITY="2" ;;
    yelp) TEST_DATA="yelp_polarity@test" &&
            VAL_DATA="yelp_polarity@dev"  &&
            TRAIN_DATA="yelp_polarity@train"  &&
            READER="HUGGINGFACE" &&
            GRANULARITY="2" ;;
    imdb) TEST_DATA="imdb@test" &&
            VAL_DATA="imdb@dev"  &&
            TRAIN_DATA="imdb@train"  &&
            READER="HUGGINGFACE" &&
            GRANULARITY="2" ;;
    semeval-2) TEST_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-test-gold.txt" &&
            VAL_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-dev.txt"  &&
            TRAIN_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-train.txt"  &&
            READER="SEMEVAL" &&
            GRANULARITY="2" ;;
    semeval-3) TEST_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-test-gold.txt" &&
        VAL_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-dev.txt"  &&
        TRAIN_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-train.txt"  &&
        READER="SEMEVAL" &&
        GRANULARITY="3" ;;
    semeval-7) TEST_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-test-gold.txt" &&
        VAL_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-dev.txt"  &&
        TRAIN_DATA="${DATA}/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-train.txt"  &&
        READER="SEMEVAL" &&
        GRANULARITY="7" ;;
    mixed) TEST_DATA="${DATA}/mixed/mixed_sst_semeval_imdb_rt_yelp_test.txt" &&
        VAL_DATA="${DATA}/mixed/mixed_sst_semeval_imdb_rt_yelp_dev.txt" &&
        TRAIN_DATA="${DATA}/mixed/mixed_sst_semeval_imdb_rt_yelp_train.txt" &&
        READER="PLAIN" &&
        GRANULARITY="2" ;;
    conll2003) TEST_DATA="${DATA}/conll2003/test.txt" &&
        VAL_DATA="${DATA}/conll2003/valid.txt" &&
        TRAIN_DATA="${DATA}/conll2003/train.txt" &&
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
echo "TEST_DATA: ${TEST_DATA}"
echo "VAL_DATA: ${VAL_DATA}"
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "exp: ${exp}"
echo "train: ${train}"
printf "\n"

cd $ROOT
source activate btools

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
        # for now allow training only on conll
        oargs="{'train_data_path': '${TRAIN_DATA}',\
                'validation_data_path': '${VAL_DATA}'\
                }"
    fi

    echo $oargs

   allennlp train "${exp}" -s "${modeldir}" -f -o "${oargs}" --include-package src.models

fi


