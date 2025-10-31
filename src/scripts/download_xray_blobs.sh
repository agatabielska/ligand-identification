#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

if [ ! -f "${data_dir}/blobs_full.tar.gz" ]; then
    echo "Downloading blobs_full.tar.gz..."
    aria2c -x 16 -s 16 -d "${data_dir}" "https://zenodo.org/records/10908325/files/blobs_full.tar.gz?download=1" || { echo "Install aria2c with 'sudo apt-get install aria2' and re-run this script"; exit 1; }
fi

tar -xvzf "${data_dir}/blobs_full.tar.gz" -C "${data_dir}"

extracted_dir="$(tar -tzf "${data_dir}/blobs_full.tar.gz" | head -1 | cut -d'/' -f1)"

if [ -d "${data_dir}/${extracted_dir}" ] && [ "${extracted_dir}" != "xray_blobs" ]; then
    mv "${data_dir}/${extracted_dir}" "${data_dir}/xray_blobs"
fi