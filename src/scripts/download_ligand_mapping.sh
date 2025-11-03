#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

if [ ! -f "${data_dir}/ligand_mapping.csv" ]; then
    echo "Downloading ligand_mapping.csv..."
    wget https://zenodo.org/records/10908325/files/ligand_mapping.csv?download=1 -O "${data_dir}/ligand_mapping.csv"
fi