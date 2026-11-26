uv run -m src.scripts.hpo --study-name hpo-spectral-UP-min_val_loss --project hpo-spectral-UP-min_val_loss --trial-epochs 25 paths.data_dir=data/unthresholded_transform_probabilistic model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TU-min_val_loss --project hpo-spectral-TU-min_val_loss --trial-epochs 25 paths.data_dir=data/thresholded_transform_uniform model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TP-min_val_loss --project hpo-spectral-TP-min_val_loss --trial-epochs 25 paths.data_dir=data/thresholded_transform_probabilistic model=spectral

uv run -m src.scripts.hpo --study-name hpo-clifford-UP-min_val_loss --project hpo-clifford-UP-min_val_loss --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/unthresholded_transform_probabilistic model=clifford
uv run -m src.scripts.hpo --study-name hpo-clifford-TU-min_val_loss --project hpo-clifford-TU-min_val_loss --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/thresholded_transform_uniform model=clifford
uv run -m src.scripts.hpo --study-name hpo-clifford-TP-min_val_loss --project hpo-clifford-TP-min_val_loss --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/thresholded_transform_probabilistic model=clifford
