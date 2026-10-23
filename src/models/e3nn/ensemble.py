"""Evaluate ensembles of checkpoints for the supported spectral protocols.

Tau selection options:
  +select=acc     highest validation accuracy (ties -> smaller tau) [default]
  +select=recall  highest validation macro recall within MAX_ACC_DROP
                   of the tau=0 validation accuracy

  uv run -m src.models.e3nn.ensemble paths.data_dir=data/unthresholded_transform_probabilistic \
    model=spectral machine.batch_size=256 machine.num_workers=8 \
      +arch=spectral +select=acc \
    '+runs=[checkpoint_dir_1,checkpoint_dir_2]'
"""
import re
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig, OmegaConf

from src.pipeline.data_loader import GPULoader, get_dataset, transform_e3nn

TAUS = (0.0, 0.25, 0.5, 0.75)
MAX_ACC_DROP = 0.02


def best_ckpt(run_dir: Path) -> Path:
    cands = []
    for p in run_dir.glob("model-*.ckpt"):
        loss, ep = re.search(r"val_loss=([0-9]+\.[0-9]+)", p.name), re.search(r"epoch=(\d+)", p.name)
        if loss and ep:
            cands.append((float(loss.group(1)), -int(ep.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"no model-*.ckpt in {run_dir}")
    return min(cands, key=lambda t: (t[0], t[1]))[2]


def get_cls(arch: str):
    if arch == "spectral":
        from src.models.e3nn.configurable_spectral import ConfigurableSpectralModel
        return ConfigurableSpectralModel
    raise ValueError(
        f"Unsupported ensemble architecture {arch!r}; use spectral"
    )


def load_model(ck: Path, device, cls):
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    for k in ("kw", "kwargs", "net_kw"):  # Lightning may keep var-keyword args nested
        if isinstance(hp.get(k), dict):
            hp.update(hp.pop(k))
    state_dict = ckpt["state_dict"]
    global_weight = state_dict.get("global_branch.1.weight")
    if global_weight is not None:
        hp.setdefault("glob_dim", global_weight.shape[0])
    model = cls(**hp)
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    tolerated_missing = {"global_mean", "global_std"}
    if missing - tolerated_missing or unexpected:
        raise RuntimeError(
            f"checkpoint structure is incompatible "
            f"(missing={sorted(missing - tolerated_missing)}, "
            f"unexpected={sorted(unexpected)})"
        )
    mismatched = [
        key for key in expected.keys() & state_dict.keys()
        if expected[key].shape != state_dict[key].shape
    ]
    if mismatched:
        raise RuntimeError(f"checkpoint tensor shapes are incompatible: {mismatched[:8]}")
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def collect_probs(models, loader):
    per_model, ys = [[] for _ in models], []
    for x, y in loader:
        for i, m in enumerate(models):
            per_model[i].append(F.softmax(m(x).float(), dim=-1))
        ys.append(y)
    return [torch.cat(p) for p in per_model], torch.cat(ys)


def predict(probs, log_prior, tau):
    return (probs.clamp_min(1e-12).log() - tau * log_prior).argmax(-1)


def evaluate(probs, y, log_prior, tau, nc):
    pred = predict(probs, log_prior, tau)
    rec = torchmetrics.Recall(num_classes=nc, average="macro", task="multiclass", zero_division=0).to(pred.device)
    return (pred == y).float().mean().item(), rec(pred, y).item()


def full_report(probs, y, log_prior, tau, nc, title=""):
    from sklearn.metrics import classification_report

    pred = predict(probs, log_prior, tau)
    print(f"\n=== {title} (tau={tau:.2f}) ===")
    print(classification_report(y.cpu().numpy(), pred.cpu().numpy(), digits=4, zero_division=0))
    k = min(5, nc)
    print(f"top-{k} acc: {(probs.topk(k, dim=-1).indices == y[:, None]).any(-1).float().mean().item():.4f}")


def choose_tau(rows, mode):
    """rows: (tau, val_acc, val_rec, test_acc, test_rec). Uses VAL columns only."""
    if mode == "acc":
        return max(rows, key=lambda r: (r[1], -r[0]))
    base = next(r for r in rows if r[0] == 0.0)
    ok = [r for r in rows if base[1] - r[1] <= MAX_ACC_DROP]
    return max(ok, key=lambda r: (r[2], -r[0]))


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    runs = list(cfg.runs)
    mode = OmegaConf.select(cfg, "select", default="acc")
    cls = get_cls(OmegaConf.select(cfg, "arch", default="spectral"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = cfg.train.out_channels

    data_dir = Path(cfg.paths.data_dir)
    classes = sorted({d.name for s in ("train", "val", "test") for d in (data_dir / s).iterdir() if d.is_dir()})
    label_to_idx = {n: i for i, n in enumerate(classes)}
    transform = transform_e3nn
    bs, nw = cfg.machine.batch_size, cfg.machine.num_workers
    val_loader = GPULoader(get_dataset(data_dir / "val", cfg, transform, label_to_idx), bs, False, device, nw)
    test_loader = GPULoader(get_dataset(data_dir / "test", cfg, transform, label_to_idx), bs, False, device, nw)

    names, models = [], []
    for r in runs:
        try:
            ck = best_ckpt(Path(cfg.paths.model_checkpoint) / r)
            models.append(load_model(ck, device, cls))
            names.append(r)
            print(f"[{r}] loaded {ck.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[{r}] SKIPPED: {type(e).__name__}: {str(e)[:150]}")
    if not models:
        raise SystemExit("no runs could be loaded")
    log_prior = models[0].log_prior

    val_probs, val_y = collect_probs(models, val_loader)
    test_probs, test_y = collect_probs(models, test_loader)

    print("\nIndividual runs (tau=0):")
    for r, vp, tp in zip(names, val_probs, test_probs):
        va, vr = evaluate(vp, val_y, log_prior, 0.0, nc)
        ta, tr = evaluate(tp, test_y, log_prior, 0.0, nc)
        print(f"  {r:28s} val acc {va:.4f} rec {vr:.4f} | test acc {ta:.4f} rec {tr:.4f}")

    ens_val, ens_test = torch.stack(val_probs).mean(0), torch.stack(test_probs).mean(0)
    print(f"\nEnsemble of {len(models)} runs:\n  tau   val_acc  val_rec | test_acc test_rec")
    rows = []
    for t in TAUS:
        va, vr = evaluate(ens_val, val_y, log_prior, t, nc)
        ta, tr = evaluate(ens_test, test_y, log_prior, t, nc)
        rows.append((t, va, vr, ta, tr))
        print(f"  {t:4.2f}  {va:.4f}   {vr:.4f}  | {ta:.4f}   {tr:.4f}")

    t, va, vr, ta, tr = choose_tau(rows, mode)
    print(f"\nTau chosen on VAL (select={mode}): {t:.2f} -> test acc {ta:.4f}, test macro recall {tr:.4f}")
    full_report(ens_val, val_y, log_prior, t, nc, title="Ensemble (VAL)")
    full_report(ens_test, test_y, log_prior, t, nc, title="Ensemble (TEST)")


if __name__ == "__main__":
    main()