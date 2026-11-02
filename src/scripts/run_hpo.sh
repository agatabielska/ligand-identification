# uv run -m src.scripts.hpo --study-name hpo-spectral-UP --project hpo-spectral-UP paths.data_dir=data/unthresholded_transform_probabilistic model=spectral
# uv run -m src.scripts.hpo --study-name hpo-spectral-TU --project hpo-spectral-TU paths.data_dir=data/thresholded_transform_uniform model=spectral
# uv run -m src.scripts.hpo --study-name hpo-spectral-TP --project hpo-spectral-TP paths.data_dir=data/thresholded_transform_probabilistic model=spectral

uv run -m src.scripts.hpo --study-name hpo-clifford-UP --project hpo-clifford-UP --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/unthresholded_transform_probabilistic model=clifford
uv run -m src.scripts.hpo --study-name hpo-clifford-TU --project hpo-clifford-TU --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/thresholded_transform_uniform model=clifford
uv run -m src.scripts.hpo --study-name hpo-clifford-TP --project hpo-clifford-TP --trial-epochs 40 --search configs/hpo/clifford.yaml paths.data_dir=data/thresholded_transform_probabilistic model=clifford