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
### Download ligand groups mapping, xray training and holdout mapping
```bash
./src/scripts/download_ligand_mapping.sh
```
### Download and unpack blobs
X-ray:
```bash
./src/scripts/download_xray_blobs.sh
```

CryoEM:
```bash
./src/scripts/download_cryoem_blobs.sh
```

### Group blobs into classes according to xray mapping (run group_xray_blobs.sh first)
X-ray:
```bash
./src/scripts/group_xray_blobs.sh
```
CryoEM:
```bash
./src/scripts/group_cryoem_blobs.sh
```


