
cd ligand-classification
mkdir data
cd data

# Download data
wget https://zenodo.org/records/10908325/files/blobs_full.tar.gz
# wget https://zenodo.org/records/10908325/files/cryoem_blobs.zip
wget https://zenodo.org/records/10908325/files/xray_holdout.csv
wget https://zenodo.org/records/10908325/files/xray_train.csv

mv xray_holdout.csv holdout.csv
mv xray_train.csv train.csv

# unzip cryoem_blobs.zip

tar -xvzf blobs_full.tar.gz

# Setup docker
cd ..
./start.sh


docker exec -it ligands-minkowski_gpu-1 bash
python generate_dataset.py