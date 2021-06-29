git submodule init
git submodule update

aws s3 cp --recursive s3://aws-comprehend-intern-data-us-east-1/czarpaul/predictions/ predictions
aws s3 cp --recursive s3://aws-comprehend-intern-data-us-east-1/czarpaul/test_suites/ test_suites

conda env create -f requirements.yml
conda activate btools

# required for plotting on ubuntuu
sudo apt-get install cm-super
