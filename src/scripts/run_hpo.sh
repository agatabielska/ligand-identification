uv run -m src.scripts.hpo --study-name hpo-spectral-UP-min_val_loss --project hpo-spectral-UP-min_val_loss --trial-epochs 25 --metric val_loss --direction minimize paths.data_dir=data/unthresholded_transform_probabilistic model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TU-min_val_loss --project hpo-spectral-TU-min_val_loss --trial-epochs 25 --metric val_loss --direction minimize paths.data_dir=data/thresholded_transform_uniform model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TP-min_val_loss --project hpo-spectral-TP-min_val_loss --trial-epochs 25 --metric val_loss --direction minimize paths.data_dir=data/thresholded_transform_probabilistic model=spectral

uv run -m src.scripts.hpo --study-name hpo-radialCNN-UP-min_val_loss --project hpo-radialCNN-UP-min_val_loss --trial-epochs 40 --metric val_loss --direction minimize --search configs/hpo/radialCNN.yaml paths.data_dir=data/unthresholded_transform_probabilistic model=radialCNN
uv run -m src.scripts.hpo --study-name hpo-radialCNN-TU-min_val_loss --project hpo-radialCNN-TU-min_val_loss --trial-epochs 40 --metric val_loss --direction minimize --search configs/hpo/radialCNN.yaml paths.data_dir=data/thresholded_transform_uniform model=radialCNN
uv run -m src.scripts.hpo --study-name hpo-radialCNN-TP-min_val_loss --project hpo-radialCNN-TP-min_val_loss --trial-epochs 40 --metric val_loss --direction minimize --search configs/hpo/radialCNN.yaml paths.data_dir=data/thresholded_transform_probabilistic model=radialCNN

uv run -m src.scripts.hpo --study-name hpo-spectral-UP --project hpo-spectral-UP --trial-epochs 25 --metric val_acc --direction maximize paths.data_dir=data/unthresholded_transform_probabilistic model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TU --project hpo-spectral-TU --trial-epochs 25 --metric val_acc --direction maximize paths.data_dir=data/thresholded_transform_uniform model=spectral
uv run -m src.scripts.hpo --study-name hpo-spectral-TP --project hpo-spectral-TP --trial-epochs 25 --metric val_acc --direction maximize paths.data_dir=data/thresholded_transform_probabilistic model=spectral

uv run -m src.scripts.hpo --study-name hpo-radialCNN-UP --project hpo-radialCNN-UP --trial-epochs 40 --metric val_acc --direction maximize --search configs/hpo/radialCNN.yaml paths.data_dir=data/unthresholded_transform_probabilistic model=radialCNN
uv run -m src.scripts.hpo --study-name hpo-radialCNN-TU --project hpo-radialCNN-TU --trial-epochs 40 --metric val_acc --direction maximize --search configs/hpo/radialCNN.yaml paths.data_dir=data/thresholded_transform_uniform model=radialCNN
uv run -m src.scripts.hpo --study-name hpo-radialCNN-TP --project hpo-radialCNN-TP --trial-epochs 40 --metric val_acc --direction maximize --search configs/hpo/radialCNN.yaml paths.data_dir=data/thresholded_transform_probabilistic model=radialCNN
