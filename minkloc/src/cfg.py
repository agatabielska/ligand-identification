# Adapted from https://github.com/jkarolczak/ligand-classification
from typing import Dict

import yaml


def read_config(path: str) -> Dict:
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    return cfg
