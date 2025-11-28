#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"

data_dir="${SCRIPT_DIR}../../data"

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

if [ ! -f "${data_dir}/blobs_full.tar.gz" ]; then
    echo "Downloading blobs_full.tar.gz..."
    url="https://zenodo.org/records/10908325/files/blobs_full.tar.gz?download=1"
    
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -d "${data_dir}" "$url" || { echo "Download failed"; exit 1; }
    else
        echo "aria2c not found, using curl instead."
        curl -# -L "$url" -o "${data_dir}/blobs_full.tar.gz" || { echo "Download failed"; exit 1; }
    fi
fi

echo "Extracting blobs_full.tar.gz..."
tar -xvzf "${data_dir}/blobs_full.tar.gz" -C "${data_dir}"

extracted_dir="$(tar -tzf "${data_dir}/blobs_full.tar.gz" | head -1 | cut -d'/' -f1)"

if [ -d "${data_dir}/${extracted_dir}" ] && [ "${extracted_dir}" != "xray_blobs" ]; then
    mv "${data_dir}/${extracted_dir}" "${data_dir}/xray_blobs"

fi

echo "Download and extraction complete!"
exit 0