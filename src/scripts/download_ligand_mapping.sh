#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

wget https://zenodo.org/records/10908325/files/ligand_mapping.csv?download=1 -O "${data_dir}/ligand_mapping.csv"