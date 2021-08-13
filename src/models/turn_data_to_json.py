# This script can be used to convert plain text sentences to json inputs --
# if one wants to use allennlp predictor on models that haven't been trained
# with plain-text reader (or a reader that can adapt to plain text reader).

import argparse
import sys
from typing import List
import json


sys.path.insert(0, f"../")
sys.path.insert(0, f".")


def is_tokenized(lines: List[str]):
    blank_lines = [x for x in lines if x == "" or x == "\n"]
    if len(blank_lines) > 1:
        return True
    else:
        return False


def main(in_file: str, out_file: str):
    with open(in_file, "r") as f:
        lines = f.readlines()

        if not is_tokenized(lines):
            with open(out_file, "w") as f2:
                for line in lines:
                    new_line = '{"sentence": "' + line.strip() + '"}\n'
                    f2.write(new_line)
        else:
            tok_sents = []
            current_tok_sent = []
            for line in lines:
                if line == "" or line == "\n":
                    if current_tok_sent:
                        tok_sents.append(current_tok_sent)
                        current_tok_sent = []
                else:
                    current_tok_sent.append(line.strip("\n"))

            if len(current_tok_sent) > 0:
                tok_sents.append(current_tok_sent)

            with open(out_file, "w") as f2:
                for ts in tok_sents:
                    tmp_d = {"tokenized_sentence": ts}
                    new_line = json.dumps(tmp_d) + "\n"
                    f2.write(new_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str)
    parser.add_argument('--out-file', type=str)
    args = parser.parse_args()
    main(args.in_file, args.out_file)
