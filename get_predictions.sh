#!/bin/bash
# all of the following are default arguments
# which can be overriden by the arguments given to this script

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" 
DATA="$ROOT/datasets"

MODELSDIR="models"

# by default the script assumes predictions are run on simple sentences
# not in the sst tree format (or other)
plain_format=1

add_neutral_class=0

# bs of size 1 ensures the right order of predictions
BATCH_SIZE=1

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
        MODELSDIR)          MODELSDIR=${VALUE} ;;  
        out_subdir)         out_subdir=${VALUE} ;;  
        BATCH_SIZE)         BATCH_SIZE=${VALUE} ;;  
        # path to the experiment file (config) or a name of the model dir
        # in MODELSDIR
        exp)                exp=${VALUE} ;; 
        # if exp not specified one can specify model_dir
        model_dir)          model_dir=${VALUE};;
        # sentences to get predictions for
        data)               data=${VALUE} ;; 
        # plain text sentences or stt trees? 0/1
        plain_format)         plain_format=${VALUE} ;;
        out_file)           out_file=${VALUE} ;;
        add_neutral_class)  add_neutral_class=${VALUE} ;;
        *)   
    esac    
done

out_subdir="$ROOT/predictions/${out_subdir}"


error=0
[ -z "$data" ] &&  error=1 &&  echo "ERROR: data argument is required, please provide a path to test sentences"
[ -z "$exp" ] && [ -z "$model_dir" ] &&  error=1 &&  echo "ERROR: exp argument is required, please provide the jsonnet config file"
[ $error == 1 ] && exit


if [ -z "$exp" ]
then
    modeldir=${model_dir}
else
    # get model name from the name of the json file
    mname=$(basename $exp)
    mname=${mname%.*}
    modeldir="$MODELSDIR/$mname"
fi



[ ! -d "$modeldir" ] && echo "Directory $modeldir doesn't exists. Cannot get predictions from such model." && exit


printf "\n>>>>>>> executing bash script with the following arguments:\n"
echo "ROOT: ${ROOT}"
echo "out_file (for checklist formatted predictions): ${out_file}"
echo "modeldir: ${modeldir}"
echo "data: ${data}"
echo "exp: ${exp}"
echo "plain_format: ${plain_format}"
echo "add_neutral_class: ${add_neutral_class}"
printf "\n"

cd $ROOT


#################################################
#       THE MAIN SCRIPT STARTS HERE             #
#################################################

dname=$(basename $data)
dname=${dname%.*}
preds_dir="${modeldir}/predictions"

mkdir -p "${preds_dir}"

predfile="${preds_dir}/${dname}.txt"

echo "Getting the predictions from model ${mname}. Output: ${predfile} ..."
if [ ${plain_format} == 1 ]
then
    if [[ "$exp" != *"ner"* ]]
    then
        allennlp predict "${modeldir}" "${data}" --output-file "${predfile}" --use-dataset-reader --batch-size "${BATCH_SIZE}" --cuda-device 0 -o "{'dataset_reader': {'reader': 'PLAIN'}, 'validation_dataset_reader':  {'reader': 'PLAIN'}}" --include-package src.models.readers --silent
    else
        # if no reader is used the predictor assumes json inputs
        echo "NO READER, SO USING PREDICTOR'S READER (PRE-TOKENIZED)"
        tmp_data=${data%.*}
        tmp_data=${tmp_data}_json_tmp.txt
        python3 src/models/turn_data_to_json.py --in-file "${data}" --out-file "${tmp_data}"
        allennlp predict "${modeldir}" "${tmp_data}" --output-file "${predfile}" --cuda-device 0 --silent --batch-size "${BATCH_SIZE}" --predictor "sentence_tagger_pjc" --include-package src.models
        rm $tmp_data
    fi
else
    allennlp predict "${modeldir}" "${data}" --output-file "${predfile}" --use-dataset-reader --batch-size "${BATCH_SIZE}" --cuda-device 0 --include-package src.models --silent
fi


if [ ! -z "$out_file" ]
then
    labels_path="${modeldir}/vocabulary/labels.txt"

    echo "Processing and formatting the predictions. Output: ${out_file} ..."
    if [ ${add_neutral_class} == 1 ]; then 
        python3 $ROOT/src/models/process_predictions.py --in-path "${predfile}" --labels-vocab-path "${labels_path}" --out-path "${out_file}" --add-neutral-class
    else
        echo "python3 $ROOT/src/models/process_predictions.py --in-path \"${predfile}\" --labels-vocab-path \"${labels_path}\" --out-path \"${out_file}\""
        python3 $ROOT/src/models/process_predictions.py --in-path "${predfile}" --labels-vocab-path "${labels_path}" --out-path "${out_file}"
    fi
fi
