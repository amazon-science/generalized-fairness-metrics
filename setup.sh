#!/bin/bash

CUDA="10.1"

if [ ! -z "$CUDA" ]; then 
    device_arg="cudatoolkit=${CUDA}"
else
    device_arg="cpuonly"
fi

echo "Setting up the environment (${device_arg}) ..."

conda env create -f requirements.yml
conda activate btools
#conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 ${device_arg} -c pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 ${device_arg} -c pytorch


conda install -c anaconda ipykernel
python -m ipykernel install --user --name=btools

# required for plotting on ubuntuu
sudo apt-get install cm-super


echo "Getting the submodules ..."
git submodule init
git submodule update

echo "Downloading the SemEval data ..."

# getting the SemEval data from
# https://competitions.codalab.org/competitions/17751#learn_the_details-datasets
wget -O semeval-train.zip http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/English/2018-Valence-oc-En-train.zip
wget -O semeval-dev.zip http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/English/2018-Valence-oc-En-dev.zip

mkdir -p datasets/semeval-v-oc
unzip semeval-train.zip -d datasets/semeval-v-oc
unzip semeval-dev.zip -d datasets/semeval-v-oc

rm semeval-dev.zip
rm semeval-train.zip

echo "DONE."

# getting CoNLL-2003 shared task data from https://www.clips.uantwerpen.be/conll2003/ner/
# wget -O conll2003.tgz https://www.clips.uantwerpen.be/conll2003/ner.tgz
# mkdir -p datasets/conll2003
# tar -xvzf conll2003.tgz -C datasets/conll2003
