#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

if [ ! -f "${data_dir}/cryoem_blobs.zip" ]; then
    echo "Downloading cryoem_blobs.zip..."
    aria2c -x 16 -s 16 -d "${data_dir}" https://zenodo.org/records/10908325/files/cryoem_blobs.zip?download=1 || { echo "Install aria2c with 'sudo apt-get install aria2' and re-run this script"; exit 1; }
fi

unzip "${data_dir}/cryoem_blobs.zip" -d "${data_dir}"