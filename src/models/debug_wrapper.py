# import json
# import shutil
# import sys
# import pathlib
# curdir = pathlib.Path(__file__).parent.absolute()
# sys.path.insert(0, f"{curdir}/../src")

# from allennlp.commands import main


# def debug():

#     config_file = "../experiments/roberta-sst-2-test.jsonnet"

#     serialization_dir = "/tmp/debugger_predict"

#     overrides = '{"dataset_reader": {"reader": "PLAIN"}}'

#     # Training will fail if the serialization directory already
#     # has stuff in it. If you are running the same training loop
#     # over and over again for debugging purposes, it will.
#     # Hence we wipe it out in advance.
#     # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
#     shutil.rmtree(serialization_dir, ignore_errors=True)

#     # Assemble the command into sys.argv
#     sys.argv = [
#         "allennlp",  # command name, not used by main
#         "predict",
#         "/home/ubuntu/czarpaul/trained_models/roberta-sst-2-test",
#         "small_test.txt",
#         "--output-file",
#         "/home/ubuntu/czarpaul/trained_models/roberta-sst-2-test/predictions/tests_n500.txt",
#         "--use-dataset-reader",
#         "--cuda-device",
#         "0",
#         "-o", overrides,
#         "--include-package", "src.models"
#     ]

#     main()