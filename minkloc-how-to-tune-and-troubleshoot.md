# MinkLoc3Dv2 tuning and troubleshooting

This runbook tunes MinkLoc3Dv2 separately on the three configured datasets,
logs metrics to W&B, and then refits the best configuration on `train + val`
before evaluating on `test`/`holdout`.

## 1. Start the GPU container

On the host, from the repository root:

```bash
./start.sh
docker ps
```

If dependencies changed, rebuild first:

```bash
docker compose --env-file docker/.env \
  -f docker/compose-gpu.yml build --no-cache minkowski_gpu
./start.sh
```

Enter the running container. Use the container name shown by `docker ps`:

```bash
docker exec -it <container-name> bash
cd /app/scripts
```

The repository is mounted at `/app`; datasets are mounted according to
`DATA_PATH` in `docker/.env`.

## 2. Configure W&B

This project uses W&B hosted by CoreWeave Forge. Inside the container:

```bash
export WANDB_BASE_URL=https://forge.coreweave.com/wandb
export WANDB_API_KEY='<your Forge W&B API key>'
wandb login --relogin --host https://forge.coreweave.com/wandb
wandb status
```

Never commit or paste the API key into source files. The current configuration
uses:

```yaml
project: ligans-hpo-1
entity: agata-bielska
```

Change these values in [`cfg/tune_sparse.yaml`](cfg/tune_sparse.yaml) if your
team uses another W&B entity or project.

## 3. Run inside tmux

Start a persistent session:

```bash
tmux new -s minkloc-tuning
cd /app/scripts
python tune_and_train_sparse.py --stage all
```

Detach without stopping the process with `Ctrl-b`, then `d`.

Reconnect later:

```bash
tmux attach -t minkloc-tuning
```

List sessions:

```bash
tmux ls
```

Stop a run by attaching and pressing `Ctrl-c`.

## 4. Run stages separately

Tune only:

```bash
python tune_and_train_sparse.py --stage tune
```

Run final refitting after tuning:

```bash
python tune_and_train_sparse.py --stage final
```

Run both stages:

```bash
python tune_and_train_sparse.py --stage all
```

The script must be run from `/app/scripts` as shown above, or from `/app`
with `python scripts/tune_and_train_sparse.py`.

## 5. Outputs

Local outputs are written under:

```text
/app/logs/hpo/
```

Important files:

| Path | Contents |
|---|---|
| `optuna.db` | Persistent Optuna studies; do not delete when resuming |
| `best_hyperparameters.json` | Best parameters and final metrics |
| `tuning.log` | Appended terminal output and progress |
| `errors.log` | Full traceback for each failed tuning trial |
| `<dataset>/trial-XXX.pt` | Best checkpoint from an HPO trial |
| `<dataset>/final.pt` | Final refitted model checkpoint |

W&B contains the per-trial and final runs, full training/validation/test
metrics for every epoch, including accuracy, top-k accuracy, recall,
precision, F1, losses, metadata, and model artifacts. Runs are grouped by
dataset and stage.

## 6. Resume after interruption

Rerun the same command:

```bash
python tune_and_train_sparse.py --stage tune
```

Optuna resumes from `logs/hpo/optuna.db`. Completed and pruned trials are not
repeated; failed trials are recorded and remaining trials continue. The
configured `trials` value is the target total number of completed/pruned
trials. The summary is updated after each dataset.

Do not delete `logs/hpo/optuna.db` or `best_hyperparameters.json` if you want
to resume.

## 7. Troubleshooting

### `python: can't open file 'scripts/tune_and_train_sparse.py'`

You are already in `/app/scripts`. Run:

```bash
python tune_and_train_sparse.py --stage all
```

### NVIDIA/CDI or NVML error

For errors such as `unresolvable CDI devices` or `Driver/library version
mismatch`, the host NVIDIA driver/container toolkit is broken. Check:

```bash
nvidia-smi
```

If it fails, repair/restart the host GPU installation before starting Docker.
The CPU fallback is:

```bash
./start.sh cpu
```

but it is not suitable for practical MinkowskiEngine training.

### W&B `401 Unauthorized` or invalid API key

Use a Forge key and endpoint:

```bash
export WANDB_BASE_URL=https://forge.coreweave.com/wandb
wandb login --relogin --host https://forge.coreweave.com/wandb
wandb status
```

Ensure the configured `entity` is an entity where the key has access.

### W&B `405 Method Not Allowed` or initialization timeout

Confirm the Forge endpoint and installed SDK:

```bash
echo "$WANDB_BASE_URL"
python -c "import wandb; print(wandb.__version__)"
```

Rebuild the GPU image after dependency changes. The project pins Optuna for
the Python 3.7 image; W&B initialization uses an extended timeout.

### Missing Optuna or tqdm

Rebuild the image, or install the declared dependencies in the active
environment:

```bash
pip install -r /app/requirements.txt
```

### Check progress without attaching

From another shell:

```bash
tail -f /app/logs/hpo/tuning.log
```

For failed trials:

```bash
tail -f /app/logs/hpo/errors.log
```

Check the process:

```bash
ps aux | grep tune_and_train_sparse
```

## 8. Configuration

Edit [`cfg/tune_sparse.yaml`](cfg/tune_sparse.yaml) to change:

- W&B project/entity
- number of trials and epochs
- datasets
- device and worker count
- optimizer and architecture search spaces

Keep `test`/`holdout` out of tuning. It is used only after refitting on
`train + val`. Training drops an incomplete final batch to avoid NetVLAD
BatchNorm failures caused by a singleton batch; validation and test retain all
examples.
