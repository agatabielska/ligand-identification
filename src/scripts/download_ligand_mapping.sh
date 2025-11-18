#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

# Create data directory if it doesn't exist
mkdir -p "${data_dir}"

# Download all files
echo "Downloading ligand_mapping.csv..."
wget https://zenodo.org/records/10908325/files/ligand_mapping.csv?download=1 -O "${data_dir}/ligand_mapping.csv"

echo "Downloading xray_holdout.csv..."
wget https://zenodo.org/records/10908325/files/xray_holdout.csv?download=1 -O "${data_dir}/xray_holdout.csv"

echo "Downloading xray_train.csv..."
wget https://zenodo.org/records/10908325/files/xray_train.csv?download=1 -O "${data_dir}/xray_train.csv"
