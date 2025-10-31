# ligand-identification

# To set up the environment
Install uv on your system if you haven't already: https://docs.astral.sh/uv/getting-started/installation/

And then run:
```bash
uv sync
```

# To run the blob visualization
```bash
uv run streamlit run src/visualizations/blob_visualization.py
```

# To prepare data for the model
### Download ligand groups mapping
```bash
./src/scripts/download_ligand_mapping.sh
```
### Download and unpack blobs
CryoEM:
```bash
./src/scripts/download_cryoem_blobs.sh
```
X-ray:
```bash
./src/scripts/download_xray_blobs.sh
```
### Group blobs into classes
CryoEM:
```bash
./src/scripts/group_cryoem_blobs.sh
```
TODO: test it
X-ray:
```bash
./src/scripts/group_xray_blobs.sh
```
### Filter out small classes
TODO: create a get_rate_groups.sh which requires unpacked, grouped ligands of both types, filter script require rare_groups.txt
TODO: change and test it
CryoEM:
```bash
./src/scripts/filter_cryoem_groups.sh
```
X-ray:
TODO: create amd test it
```bash
./src/scripts/filter_xray_groups.sh
```