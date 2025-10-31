#!/usr/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

aria2c -x 16 -s 16 -d "${data_dir}" https://zenodo.org/records/10908325/files/cryoem_blobs.zip?download=1

unzip "${data_dir}/cryoem_blobs.zip" -d "${data_dir}"