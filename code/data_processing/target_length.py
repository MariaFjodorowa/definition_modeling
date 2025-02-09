import argparse
import csv

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/scratch/project_465001384/corpora/defgen_data/train_axolotl24st_fi.tsv.gz')
    parser.add_argument('--tokenizer', default='CohereForAI/aya-101')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.data)
    definitions = pd.read_csv(
        args.data,
        sep='\t',
        compression='gzip',
        quoting=csv.QUOTE_NONE,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    definitions['def_length'] = definitions.definition.apply(lambda x: len(tokenizer.tokenize(x)))
    print(definitions.def_length.mean())
