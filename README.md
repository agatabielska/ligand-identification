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

# NEW, CryoEM only class grouping
```bash
./src/scripts/download_cryoem_blobs.sh
uv run -m src.scripts.preprocess --raw-data-dir "cryoem_blobs" --output-folder "data/thresholded" --transform-stack "probabilistic"
```
To remove unnecessary files from NoThresholdBlobs run:
```bash
uv run -m src.scripts.trim_unthresholded
```
And then to filter the classes:
```bash
uv run -m src.scripts.preprocess --raw-data-dir "NoThresholdBlobs" --output-folder "data/unthresholded" --transform-stack "probabilistic"
```

Resulting groupings will be in `data/filtered_cryoem_classes/`
