#!/usr/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/"
data_dir="${SCRIPT_DIR}../../data"

if [ ! -d "${data_dir}" ]; then
    mkdir -p "${data_dir}"
fi

if [ ! -f "${data_dir}/cryoem_blobs.zip" ]; then
    echo "Downloading cryoem_blobs.zip..."
    url="https://zenodo.org/records/10908325/files/cryoem_blobs.zip?download=1"
    
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -d "${data_dir}" "$url"  || { echo "Download failed"; exit 1; }
    else
        echo "aria2c not found, using curl instead."
        curl -# -L "$url" -o "${data_dir}/cryoem_blobs.zip" || { echo "Download failed"; exit 1; }
    fi
fi

echo "Extracting cryoem_blobs.zip..."
unzip -q "${data_dir}/cryoem_blobs.zip" -d "${data_dir}" || { echo "Extraction failed"; exit 1; }

echo "Download and extraction complete!"
exit 0