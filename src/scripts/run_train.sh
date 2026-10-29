#!/bin/bash
#SBATCH -p hgx
#SBATCH -w hgx2
#SBATCH -n1
#SBATCH -A inz_cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 64
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/slurm-%j.out
##SBATCH --exclusive
#SBATCH --cpus-per-task=16

# uv run -m src.scripts.train paths.data_dir=data/processed/cryoem_prob model=e3nn
# uv run -m src.scripts.train paths.data_dir=data/processed/cryoem_unif model=e3nn

# uv run -m src.scripts.train paths.data_dir=data/processed/cryoem_prob model=clifford
# uv run -m src.scripts.train paths.data_dir=data/processed/cryoem_unif model=clifford


# uv run -m src.scripts.train paths.data_dir=data/thresholded_transform_probabilistic model=clifford wandb.run_name=test
# uv run -m src.scripts.train paths.data_dir=data/thresholded_transform_probabilistic model=clifford wandb.run_name=thresholded_probabilistic_clifford_flat_input_scalarshell_fixed
# uv run -m src.scripts.train paths.data_dir=data/thresholded_transform_uniform model=clifford wandb.run_name=thresholded_uniform_clifford_flat_input_scalarshell_fixed
# uv run -m src.scripts.train paths.data_dir=data/unthresholded_transform_probabilistic model=clifford wandb.run_name=unthresholded_probabilistic_clifford_flat_input_scalarshell_fixed


uv run -m src.scripts.train paths.data_dir=data/unthresholded_transform_probabilistic model=spectral wandb.run_name=UP_SOTA