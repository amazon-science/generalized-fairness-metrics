## Generalized Fairness Metrics

This repository contains the source code for the paper:

> [Quantifying Social Biases in NLP: A Generalization and Empirical Comparison of Extrinsic Fairness Metrics](https://arxiv.org/abs/2106.14574)\
> Paula Czarnowska, Yogarshi Vyas, Kashif Shah\
> Transaction of the Association for Computational Linguistics (TACL), 2021


__Reproducing classification experiments__:

1. Change the *MODELSDIR* variable in *get\_predictions.sh* and the *OUTDIR* variable in *run_experiments.sh* to where your models will be/are saved.

2. Change the CUDA variable in *setup.sh* to the appropriate version of CUDA.

3. Run *setup.sh* to:
    * fetch the required submodules
    * create and activate a new environment named *btools* based on the requirements.yml
    * download the SemEval valence classification data

4. Train the models from the config files in the *experiments* directory:
    > ./run_experiment.sh train=1 DATASET=semeval-2 exp=experiments/roberta.jsonnet\
    > ./run_experiment.sh train=1 DATASET=semeval-3 exp=experiments/roberta.jsonnet

5. Create the test suites and test the models. The plots for the results are saved in the *plots* directory:
    > conda activate btools\
    > python3 reproduce.py --classification --create-tests


__Reproducing NER experiments__:

1. Run the setup steps (1 and 2 above).

2. Get the CoNLL2003 data (https://www.clips.uantwerpen.be/conll2003/ner/). 
Place the *eng.train, eng.testa* and *eng.testb* files in *datasets/conll2003/ner* directory.

3. Train the model:
    > ./run_experiment.sh train=1 DATASET=conll2003 exp=experiments/ner-roberta.jsonnet

4. Test the trained model:
    > python3 reproduce.py --ner

    or, if you haven't created the test suites yet:
    > python3 reproduce.py --ner --create-tests


__Metric implementations__:

Implementations of all metrics can be found in *expanded_checklist/checklist/tests*.\
The code for generalized metrics is located in *expanded_checklist/checklist/tests/abstract_tests/generalized_metrics.py*.

## Acknowledgements
The code in the expanded_checklist directory is a restructured and expanded version of the repository 
> https://github.com/marcotcr/checklist

containing the code for testing NLP Models as described in the following paper:
> [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://aclanthology.org/2020.acl-main.442/)\
> Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh\
> Association for Computational Linguistics (ACL), 2020

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
