"""Unified spectral experiment runner.

This module combines the reusable cached BigNet components and the current
strict 2,000-point runner. The public CLI runs the current no-cache protocol:

    uv run -m src.experiments.spectral_experiments --force

It computes features from the 2,000-point tensors, trains the selected
presets, restores the lowest-validation-loss checkpoint, and reports
validation-selected ensemble pools.
"""
import argparse
import math
import statistics as st
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # wandb is optional: runs still work, just without logging
    wandb = None

from src.models.e3nn.spectral_net import SpectralNet
def macro_recall(pred, target, num_classes):
    recalls = []
    for cls in range(num_classes):
        mask = target == cls
        recalls.append((pred[mask] == cls).float().mean() if mask.any() else pred.new_tensor(0.0))
    return torch.stack(recalls).mean().item()


def select(features, meta, channels, lmax):
    del meta, channels, lmax
    return features

# Permit TensorFloat-32 matrix multiplies where supported; this favors GPU
# throughput and does not change the point representation.
torch.backends.cuda.matmul.fp32_precision = "ieee"
# torch.backends.cuda.matmul.fp32_precision = "tf32"

E3NN_LMAX_MAX = 11  # e3nn.o3.spherical_harmonics is implemented only up to l = 11
WANDB_PROJECT = "ligand-identification"  # same project as train.py
TAUS = (0.0, 0.25, 0.5, 0.75)
MAX_ACC_DROP = 0.02
GLOB = {"none": [], "all": list(range(12)), "geom": list(range(3, 12)), "count": [0, 1, 2]}
# size scalars ("all" 12 global features) are on by default: they clearly helped in group A
TRAIN_DEFAULTS = dict(lr=3e-3, wd=5e-2, epochs=15, bs=256, label_smoothing=0.05, glob="all", glob_noise=0.0)


# ----------------------------------------------------------------------------- model
class BigNet(SpectralNet):
    """SpectralNet + optional global-scalar branch + arbitrary MLP head (widths)."""

    def __init__(self, num_classes, n_glob=0, glob_dim=64, widths=(512, 256), head_dropout=0.2,
                 feat_dropout=0.0, norm="layer", **kw):
        for k in ("hidden", "n_hidden", "dropout"):
            kw.pop(k, None)
        super().__init__(num_classes, norm=norm, n_hidden=0, **kw)  # parent head is replaced below
        self.n_glob, self.d_feat = n_glob, self.d_in
        d = self.d_in
        if n_glob:
            self.glob = nn.Sequential(nn.BatchNorm1d(n_glob), nn.Linear(n_glob, glob_dim), nn.GELU())
            d += glob_dim
        self.d_in = d
        layers = [nn.LayerNorm(d) if norm == "layer" else nn.BatchNorm1d(d)]
        if feat_dropout > 0:
            layers.append(nn.Dropout(feat_dropout))
        prev = d
        for i, w in enumerate(widths):
            layers += [nn.Linear(prev, w), nn.GELU()]
            if len(widths) == 1 or i < len(widths) - 1:
                layers.append(nn.Dropout(head_dropout))  # same layout as SpectralNet for widths=(512, 256)
            prev = w
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, c, g=None):
        with torch.autocast(device_type=c.device.type, enabled=False):
            f = self.features(c.float())
            if self.n_glob:
                f = torch.cat([f, self.glob(g.float())], -1)
        return self.head(f)


# ----------------------------------------------------------------------------- presets
# NET0 = best small-scale config (v3_ls05): 4 shells, 2 channels, no Gram, n_mix 16, no dead triples,
# 2 hidden layers (512, 256), per-sample LayerNorm on features, lr 3e-3, label smoothing 0.05.
NET0 = dict(n_mix=16, gram="none", triples="nonzero", n_channels=2, in_bn=False, feat_dropout=0.0,
            bispec_mode="batched", widths=(512, 256), head_dropout=0.2, norm="layer")


def P(cache="feats4.pt", net=None, **train):
    return dict(cache=cache, net={**NET0, **(net or {})}, train=train)


SP = dict(bispec_mode="sparse")  # batched pads to (2L+1)^2 per triple: too much memory above lmax 8
PRESETS = {
    # reference: the old default, WITHOUT size scalars
    "b0":        P(glob="none"),
    # A. size scalars (scale_probe: alone 0.687 val acc); g_all == the new default
    "g_all":     P(glob="all"),
    "g_geom":    P(glob="geom"),
    "g_all_n":   P(glob="all", glob_noise=0.1),               # + noise on the scalars (anti-overfit)
    "g_all_bn":  P(net=dict(norm="batch"), glob="all"),       # fallback: per-feature BN so the scalars cannot drown the spectral features
    # B. angular resolution, bispectrum up to the same degree (needs feats4_l8/l10/l11.pt; e3nn caps l at 11)
    "l8":        P("feats4_l8.pt", dict(bispec_lmax=8)),
    "l10":       P("feats4_l10.pt", dict(bispec_lmax=10, **SP)),
    "l11":       P("feats4_l11.pt", dict(bispec_lmax=11, **SP), bs=128, lr=2e-3),
    "l11_b8":    P("feats4_l11.pt", dict(bispec_lmax=8)),       # power spectrum to 11, bispectrum only to 8
    "l10_b6":    P("feats4_l10.pt", dict(bispec_lmax=6)),
    # C. wider mix (diag power keeps the feature count sane)
    "m32d":      P(net=dict(n_mix=32, pow_mode="diag")),
    "m64d":      P(net=dict(n_mix=64, pow_mode="diag")),
    "m32t":      P(net=dict(n_mix=32)),                          # full cross-channel power, for contrast
    # D. head capacity (small-scale: 3.8M -> 0.8M params cost nothing; is the reverse also free?)
    "h_wide":    P(net=dict(widths=(2048, 1024), head_dropout=0.3)),
    "h_deep":    P(net=dict(widths=(1024, 1024, 512), head_dropout=0.3)),
    "h_huge":    P(net=dict(widths=(4096, 2048, 1024), head_dropout=0.4, feat_dropout=0.1)),
    # E. combos: edit after A-D tell you what helps. Gentler lr / more epochs for the bigger ones.
    "e_mega10":  P("feats4_l10.pt", dict(bispec_lmax=10, n_mix=32, pow_mode="diag", widths=(1024, 512),
                                         head_dropout=0.3, **SP), glob="all", lr=2e-3, epochs=20),
    "e_mega11":  P("feats4_l11.pt", dict(bispec_lmax=11, n_mix=32, pow_mode="diag", widths=(1024, 512),
                                         head_dropout=0.3, **SP), lr=1.5e-3, epochs=20, bs=128),
}
for _n, _p in PRESETS.items():  # fail at import, not after an hour of caching / training
    _l = max(int(_p["cache"].split("_l")[1].split(".")[0]) if "_l" in _p["cache"] else 6,
             _p["net"].get("bispec_lmax", 0), _p["net"].get("lmax", 0))
    assert _l <= E3NN_LMAX_MAX, f"preset {_n}: degree {_l} > e3nn limit {E3NN_LMAX_MAX}"

GROUPS = {
    "A": ["g_all", "g_geom", "g_all_n", "g_all_bn"],  # expand() adds the b0 (no scalars) reference
    "B": ["l8", "l10", "l11", "l11_b8", "l10_b6"],
    "C": ["m32d", "m64d", "m32t"],
    "D": ["h_wide", "h_deep", "h_huge"],
    "E": ["e_mega10", "e_mega11"],
    "ref": ["b0"],
}


def expand(arg):
    out = []
    for t in arg.split(","):
        out += (["b0"] + GROUPS[t] if t == "A" else GROUPS[t]) if t in GROUPS else [t]
    return list(dict.fromkeys(out))


# ----------------------------------------------------------------------------- data
def load_data(cache_path, glob_path, dev):
    d = torch.load(cache_path, map_location="cpu")
    g = torch.load(glob_path, map_location="cpu")
    for k in ("train", "val", "test"):
        assert torch.equal(d[k + "_y"].cpu(), g[k + "_y"].cpu()), f"label order of {cache_path} != {glob_path} ({k})"
    mu, sd = g["train"].mean(0), g["train"].std(0).clamp(min=1e-6)
    for k in ("train", "val", "test"):
        d[k] = d[k].to(dev)
        d["g_" + k] = ((g[k] - mu) / sd).to(dev)
    d["meta"].setdefault("r_max", 1.0)
    return d


@torch.no_grad()
def predict(net, X, G, cols, sel, bs, dev):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        g = G[i:i + bs][:, cols] if len(cols) else None
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(net(sel(X[i:i + bs]), g).float())
    return torch.cat(out)


def metrics(lg, y, nc):
    return dict(loss=F.cross_entropy(lg, y).item(), acc=(lg.argmax(1) == y).float().mean().item(),
                rec=macro_recall(lg.argmax(1), y, nc))


def run_job(spec, seed, data, dev, name="", use_wandb=False, wandb_project=WANDB_PROJECT):
    meta, nc = data["meta"], data["meta"]["num_classes"]
    net_kw, tc = dict(spec["net"]), {**TRAIN_DEFAULTS, **spec["train"]}
    C, lmax = net_kw.get("n_channels", meta["n_channels"]), net_kw.get("lmax", meta["lmax"])
    net_kw.update(n_channels=C, lmax=lmax)
    assert max(lmax, net_kw.get("bispec_lmax", 0)) <= E3NN_LMAX_MAX, \
        f"lmax {lmax} / bispec_lmax {net_kw.get('bispec_lmax')} exceeds the e3nn limit {E3NN_LMAX_MAX}"
    sel = lambda c: select(c, meta, C, lmax)
    cols = torch.tensor(GLOB[tc["glob"]], dtype=torch.long, device=dev)

    torch.manual_seed(seed)
    net = BigNet(nc, n_shells=meta["n_shells"], r_max=meta["r_max"], n_glob=len(cols), **net_kw).to(dev)
    tr, ytr, gtr = data["train"], data["train_y"].to(dev), data["g_train"]
    yv, yt = data["val_y"].to(dev), data["test_y"].to(dev)
    copies, N = tr.shape[0], tr.shape[1]
    counts = torch.bincount(ytr, minlength=nc).float()
    log_prior = ((counts + 1) / (counts.sum() + nc)).log()

    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    total = tc["epochs"] * ((N + tc["bs"] - 1) // tc["bs"])
    warm = max(1, int(total * 0.03))
    lam = lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    run = None
    if use_wandb and wandb is not None:
        run = wandb.init(project=wandb_project, name=f"{name}__s{seed}", group=name, job_type="big_runs",
                         tags=["big_runs", name], reinit=True,
                         config=dict(preset=name, seed=seed, cache=spec["cache"], net=net_kw, train=tc,
                                     n_shells=meta["n_shells"], num_classes=nc, n_glob=len(cols)))

    best, t_train = None, 0.0
    for ep in range(tc["epochs"]):
        net.train()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        perm = torch.randperm(N, device=dev)
        tl, ta, tn = torch.zeros((), device=dev), torch.zeros((), device=dev), 0  # no per-step host sync
        for i in range(0, N, tc["bs"]):
            idx = perm[i:i + tc["bs"]]
            if len(idx) < 2:
                continue
            ci = torch.randint(copies, (len(idx),), device=dev)
            c = sel(tr[ci, idx])
            g = None
            if len(cols):
                g = gtr[idx][:, cols]
                if tc["glob_noise"] > 0:
                    g = g + tc["glob_noise"] * torch.randn_like(g)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = net(c, g).float()
            loss = F.cross_entropy(logits, ytr[idx], label_smoothing=tc["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.detach() * len(idx)
            ta += (logits.argmax(1) == ytr[idx]).sum()
            tn += len(idx)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        v_lg = predict(net, data["val"], data["g_val"], cols, sel, tc["bs"], dev)
        v = metrics(v_lg, yv, nc)
        log = dict(epoch=ep, train_loss=tl.item() / max(tn, 1), train_acc=ta.item() / max(tn, 1),
                   val_loss=v["loss"], val_acc=v["acc"], val_macro_recall=v["rec"],
                   lr=opt.param_groups[0]["lr"], epoch_time_s=time.time() - t0)
        if best is None or v["loss"] < best["val_loss"]:  # checkpoint = lowest VAL loss
            t_lg = predict(net, data["test"], data["g_test"], cols, sel, tc["bs"], dev)
            t = metrics(t_lg, yt, nc)
            best = dict(epoch=ep, val_acc=v["acc"], val_loss=v["loss"], val_rec=v["rec"],
                        test_acc=t["acc"], test_loss=t["loss"], test_rec=t["rec"],
                        val_logits=v_lg.cpu(), test_logits=t_lg.cpu())
            log.update(test_loss=t["loss"], test_acc=t["acc"], test_macro_recall=t["rec"], is_best=1)
            if run is not None:  # test metrics at the best-val-loss epoch, like trainer.test(ckpt_path="best")
                run.summary.update(dict(best_epoch=ep, val_acc=v["acc"], val_loss=v["loss"], val_macro_recall=v["rec"],
                                        test_acc=t["acc"], test_loss=t["loss"], test_macro_recall=t["rec"]))
        if run is not None:
            run.log(log, step=ep)
    best.update(params=sum(p.numel() for p in net.parameters()), d_in=net.d_in,
                s_per_epoch=t_train / tc["epochs"], log_prior=log_prior.cpu(),
                val_y=data["val_y"].cpu(), test_y=data["test_y"].cpu())
    if run is not None:
        run.summary.update(dict(params=best["params"], d_in=best["d_in"], s_per_epoch=best["s_per_epoch"]))
        run.finish()
    return best


# ----------------------------------------------------------------------------- reporting
def ensemble(runs, select_rule):
    nc = runs[0]["log_prior"].numel()
    yv, yt, lp = runs[0]["val_y"], runs[0]["test_y"], runs[0]["log_prior"]
    vp = torch.stack([F.softmax(r["val_logits"], -1) for r in runs]).mean(0)
    tp = torch.stack([F.softmax(r["test_logits"], -1) for r in runs]).mean(0)
    rows = []
    for t in TAUS:
        pv = (vp.clamp_min(1e-12).log() - t * lp).argmax(-1)
        pt = (tp.clamp_min(1e-12).log() - t * lp).argmax(-1)
        rows.append((t, (pv == yv).float().mean().item(), macro_recall(pv, yv, nc),
                     (pt == yt).float().mean().item(), macro_recall(pt, yt, nc)))
    if select_rule == "acc":
        chosen = max(rows, key=lambda r: (r[1], -r[0]))
    else:
        ok = [r for r in rows if rows[0][1] - r[1] <= MAX_ACC_DROP]
        chosen = max(ok, key=lambda r: (r[2], -r[0]))
    return rows, chosen


def ms(xs):
    return f"{st.mean(xs):.4f}±{(st.stdev(xs) if len(xs) > 1 else 0):.4f}"


def report(names, out_dir, select_rule):
    runs = {n: [torch.load(p) for p in sorted(out_dir.glob(f"{n}__s*.pt"))] for n in names}
    runs = {n: r for n, r in runs.items() if r}
    if not runs:
        print("no saved runs found in", out_dir)
        return {}, {}
    hdr = (f"{'preset':10s} {'n':>2s} {'params(M)':>9s} {'d_in':>6s} {'s/ep':>5s} | {'val acc':>13s} {'val loss':>13s} | "
           f"{'test acc':>13s} {'test rec':>13s} | ens: tau val_acc | test_acc test_rec")
    print("\n" + hdr)
    ens_val, summ = {}, {}
    for n, rs in sorted(runs.items(), key=lambda kv: -st.mean(r["val_acc"] for r in kv[1])):
        _, ch = ensemble(rs, select_rule)
        ens_val[n] = ch[1]
        g = lambda k: [r[k] for r in rs]
        summ[n] = dict(n=len(rs), params_M=rs[0]["params"] / 1e6, d_in=rs[0]["d_in"], s_per_epoch=st.mean(g("s_per_epoch")),
                       val_acc=st.mean(g("val_acc")), val_loss=st.mean(g("val_loss")), test_acc=st.mean(g("test_acc")),
                       test_rec=st.mean(g("test_rec")), ens_tau=ch[0], ens_val_acc=ch[1], ens_test_acc=ch[3],
                       ens_test_rec=ch[4])
        print(f"{n:10s} {len(rs):2d} {rs[0]['params'] / 1e6:9.2f} {rs[0]['d_in']:6d} {st.mean(g('s_per_epoch')):5.1f} | "
              f"{ms(g('val_acc')):>13s} {ms(g('val_loss')):>13s} | {ms(g('test_acc')):>13s} {ms(g('test_rec')):>13s} | "
              f"     {ch[0]:.2f} {ch[1]:.4f}  | {ch[3]:.4f}   {ch[4]:.4f}")
    pools = {"POOL all": [r for rs in runs.values() for r in rs]}
    top = sorted(ens_val, key=ens_val.get, reverse=True)[:3]  # chosen by VAL ensemble accuracy
    pools[f"POOL top3 ({'+'.join(top)})"] = [r for n in top for r in runs[n]]
    pool_res = {}
    for name, rs in pools.items():
        rows, ch = ensemble(rs, select_rule)
        pool_res[name] = dict(n_models=len(rs), rows=rows, chosen_tau=ch[0])
        print(f"\n{name}: {len(rs)} models")
        print("   tau  val_acc val_rec | test_acc test_rec")
        for t, va, vr, ta, tr in rows:
            print(f"  {t:4.2f}  {va:.4f}  {vr:.4f} | {ta:.4f}   {tr:.4f}{'   <- chosen on VAL' if t == ch[0] else ''}")
    return summ, pool_res


def log_report_wandb(summ, pool_res, a):
    """One extra wandb run with the per-preset table and the ensemble/pool tables."""
    if wandb is None or not summ:
        return
    run = wandb.init(project=a.wandb_project, name="big_runs_report", group="report", job_type="report",
                     tags=["big_runs", "report"], reinit=True, config=vars(a))
    cols = ["preset"] + list(next(iter(summ.values())).keys())
    run.log({"presets": wandb.Table(columns=cols, data=[[n, *d.values()] for n, d in summ.items()])})
    for name, d in pool_res.items():
        t = wandb.Table(columns=["tau", "val_acc", "val_rec", "test_acc", "test_rec", "chosen_on_val"],
                        data=[[*r, r[0] == d["chosen_tau"]] for r in d["rows"]])
        run.log({f"pool/{name}": t})
        run.summary[f"pool/{name}/n_models"] = d["n_models"]
    run.finish()
import copy
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf  # noqa: F401

from src.models.e3nn.features import sh_coefficients
from src.pipeline.data_loader import MAX_POINTS, build_data
from src.pipeline.global_statistics import (
    complete_input_statistics_for_splits,
    point_statistics,
)
from src.utils.set_seed import set_seed

# ----------------------------------------------------------------------------- strict 2000-point protocol
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = "data/unthresholded_transform_probabilistic"
OUT_DIR = ROOT / "results" / "best_e2e"
GLOBALS_REF = ROOT / "cache" / "globals.pt"
N_SHELLS, LMAX, R_MAX = 4, 6, 1.0
POINT_DROPOUT, DENSITY_NOISE = 0.1, 0.0
P_CLEAN = 1 / 8

# ----------------------------------------------------------------------------- raw 2000-point protocol
# Only things that worked in the big run, plus combos of them. Everything starts from the winner
# (g_all_bn = NET0 + size scalars + per-feature BatchNorm). Dropped: lmax 8/10/11 (no gain, slower,
# worse val loss), e_mega*, h_huge (17M params, no gain), b0 (no scalars), raw g_all (LayerNorm).
BN = dict(norm="batch")
M32D = dict(n_mix=32, pow_mode="diag")
DEEP = dict(widths=(1024, 1024, 512), head_dropout=0.3)

CONFIGS = {
    # name:            (net overrides,                        train overrides)
    "r1_gall_bn":      ({**BN},                               {}),
    "r2_bn_m32d":      ({**BN, **M32D},                       {}),                    # best val loss in group C
    "r3_bn_m32t":      ({**BN, "n_mix": 32},                  {}),                    # m32t: 2nd best val acc
    "r4_bn_m64d":      ({**BN, "n_mix": 64, "pow_mode": "diag"}, {}),
    "r5_bn_deep":      ({**BN, **DEEP},                       {}),                    # h_deep: best test acc of group D
    "r6_bn_noise":     ({**BN},                               {"glob_noise": 0.1}),   # g_all_n idea on top of BN
    "r7_bn_m32d_deep": ({**BN, **M32D, **DEEP},               {}),
    "r8_bn_m32t_deep_noise": ({**BN, "n_mix": 32, **DEEP},    {"glob_noise": 0.1}),
}


# ----------------------------------------------------------------------------- data (raw, no cache)
def load_raw():
    """Load normalized point tensors with shape ``(N, 2000, 4)``."""
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="config", overrides=[
            f"paths.data_dir={DATA_DIR}", "model=spectral", f"model.n_shells={N_SHELLS}",
            f"model.lmax={LMAX}", "machine.batch_size=256", "machine.num_workers=8"])
    set_seed(cfg.random_seed)
    tr, va, te = build_data(cfg, "spectral")
    for name, ld in (("train", tr), ("val", va), ("test", te)):
        assert ld.x.shape[1] == MAX_POINTS, f"{name}: {ld.x.shape[1]} points, expected {MAX_POINTS}"
    return tr, va, te, int(cfg.train.out_channels)


@torch.no_grad()
def point_globals(x: torch.Tensor, bs: int = 1024) -> torch.Tensor:
    """Compute winner globals from the same normalized 2000-point tensors as the model."""
    return point_statistics(x, batch_size=bs)
def augment(x, point_dropout=POINT_DROPOUT, density_noise=DENSITY_NOISE, clean_probability=P_CLEAN):
    """Same ops as cache_features.augment, but sampled fresh per sample and per step."""
    pd, dn = point_dropout, density_noise
    xa = x
    if pd > 0:
        keep = (torch.rand_like(xa[..., 3]) > pd).to(xa.dtype)
        xa = torch.cat([xa[..., :3], (xa[..., 3] * keep).unsqueeze(-1)], -1)
    if dn > 0:
        rho = xa[..., 3] * torch.exp(torch.randn_like(xa[..., 3]) * dn)
        xa = torch.cat([xa[..., :3], rho.unsqueeze(-1)], -1)
    clean = torch.rand(x.shape[0], device=x.device) < clean_probability
    return torch.where(clean[:, None, None], x, xa)


def coeffs(x, C, lmax=LMAX, n_shells=N_SHELLS, r_max=R_MAX, chunk=1024):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return sh_coefficients(x.float(), lmax=lmax, n_shells=n_shells, r_max=r_max, n_channels=C, chunk=chunk)


@torch.no_grad()
def predict(net, X, G, cols, C, bs, dev, lmax=LMAX, n_shells=N_SHELLS, r_max=R_MAX):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        c = coeffs(X[i:i + bs].to(dev), C, lmax=lmax, n_shells=n_shells, r_max=r_max)
        g = G[i:i + bs].to(dev)[:, cols] if len(cols) else None
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(net(c, g).float())
    return torch.cat(out)


# ----------------------------------------------------------------------------- one run
def run_job(name, net_ov, train_ov, seed, D, dev, epochs=None,
            n_shells=N_SHELLS, lmax=LMAX, r_max=R_MAX,
            point_dropout=POINT_DROPOUT, density_noise=DENSITY_NOISE,
            clean_probability=P_CLEAN):
    nc = D["nc"]
    tc = {**TRAIN_DEFAULTS, **train_ov}
    if epochs:
        tc["epochs"] = epochs
    net_kw = {**NET0, **net_ov}
    C = net_kw["n_channels"]
    net_kw.update(lmax=lmax)
    cols = torch.tensor(GLOB[tc["glob"]], dtype=torch.long, device=dev)

    torch.manual_seed(seed)
    net = BigNet(nc, n_shells=n_shells, r_max=r_max, n_glob=len(cols), **net_kw).to(dev)
    Xtr, ytr, Gtr = D["tr_x"], D["tr_y"].to(dev), D["g_tr"]
    yv, yt = D["va_y"].to(dev), D["te_y"].to(dev)
    N = len(Xtr)
    counts = torch.bincount(ytr, minlength=nc).float()
    log_prior = ((counts + 1) / (counts.sum() + nc)).log()

    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    total = tc["epochs"] * ((N + tc["bs"] - 1) // tc["bs"])
    warm = max(1, int(total * 0.03))
    lam = lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    best, best_state, t_train = None, None, 0.0
    for ep in range(tc["epochs"]):
        net.train()
        t0 = time.time()
        perm = torch.randperm(N)
        tl, ta, tn = 0.0, 0, 0
        for i in range(0, N, tc["bs"]):
            idx = perm[i:i + tc["bs"]]
            if len(idx) < 2:
                continue
            x = augment(Xtr[idx].to(dev, non_blocking=True), point_dropout, density_noise, clean_probability)
            c = coeffs(x, C, lmax=lmax, n_shells=n_shells, r_max=r_max)  # 2000-point features
            g = None
            if len(cols):
                g = Gtr[idx].to(dev)[:, cols]
                if tc["glob_noise"] > 0:
                    g = g + tc["glob_noise"] * torch.randn_like(g)
            y = ytr[idx.to(dev)]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = net(c, g).float()
            loss = F.cross_entropy(logits, y, label_smoothing=tc["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            ta += (logits.argmax(1) == y).sum().item()
            tn += len(idx)
        t_train += time.time() - t0
        v_lg = predict(net, D["va_x"], D["g_va"], cols, C, tc["bs"], dev, lmax, n_shells, r_max)
        v = metrics(v_lg, yv, nc)
        print(f"  {name} s{seed} ep {ep + 1:2d}/{tc['epochs']} | train loss {tl / tn:.4f} acc {ta / tn:.4f} | "
              f"val loss {v['loss']:.4f} acc {v['acc']:.4f} rec {v['rec']:.4f} | {time.time() - t0:.0f}s", flush=True)
        if best is None or v["loss"] < best["val_loss"]:      # checkpoint = lowest VAL loss (as in big_runs)
            best = dict(epoch=ep, val_acc=v["acc"], val_loss=v["loss"], val_rec=v["rec"], val_logits=v_lg.cpu())
            best_state = copy.deepcopy(net.state_dict())

    # ---- TEST: exactly once, on the best-val-loss weights
    net.load_state_dict(best_state)
    t_lg = predict(net, D["te_x"], D["g_te"], cols, C, tc["bs"], dev, lmax, n_shells, r_max)
    t = metrics(t_lg, yt, nc)
    best.update(test_acc=t["acc"], test_loss=t["loss"], test_rec=t["rec"], test_logits=t_lg.cpu(),
                params=sum(p.numel() for p in net.parameters()), d_in=net.d_in,
                s_per_epoch=t_train / tc["epochs"], log_prior=log_prior.cpu(),
                val_y=D["va_y"].cpu(), test_y=D["te_y"].cpu())
    return best


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed_offset", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", default="", help="comma list of config names; empty runs all configs")
    ap.add_argument("--no_check", action="store_true", help="skip comparing computed globals with cache/globals.pt")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        names = a.only.split(",") if a.only else list(CONFIGS)
        report(names, a.output_dir, "acc")
        return
    names = [n for n in CONFIGS if not a.only or n in a.only.split(",")]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr, va, te, nc = load_raw()
    print(f"split sizes: train {len(tr.y)}, val {len(va.y)}, test {len(te.y)}, classes {nc}", flush=True)

    # Globals are deliberately computed from the same 2000-point tensors as spectral features.
    g = {k: point_globals(ld.x) for k, ld in (("tr", tr), ("va", va), ("te", te))}
    mu, sd = g["tr"].mean(0), g["tr"].std(0).clamp(min=1e-6)
    D = dict(nc=nc, tr_x=tr.x, va_x=va.x, te_x=te.x, tr_y=tr.y.cpu(), va_y=va.y.cpu(), te_y=te.y.cpu(),
             g_tr=(g["tr"] - mu) / sd, g_va=(g["va"] - mu) / sd, g_te=(g["te"] - mu) / sd)

    for name in names:
        net_ov, train_ov = CONFIGS[name]
        for s in range(a.seeds):
            seed = a.seed_offset + s
            f = OUT_DIR / f"{name}__s{seed}.pt"
            if f.exists() and not a.force:
                print(f"skip {f.name} (exists)")
                continue
            r = run_job(name, net_ov, train_ov, seed, D, dev, a.epochs)
            torch.save({**r, "preset": name, "seed": seed}, f)
            print(f"{name:22s} seed {seed}: ep {r['epoch'] + 1:2d} | val acc {r['val_acc']:.4f} loss {r['val_loss']:.4f} | "
                  f"test acc {r['test_acc']:.4f} rec {r['test_rec']:.4f} | {r['params'] / 1e6:.2f}M params, "
                  f"d_in {r['d_in']}, {r['s_per_epoch']:.1f}s/ep", flush=True)
    report(names, OUT_DIR, "acc")   # per-config table + POOL all + POOL top3, tau chosen on VAL


def run_default_from_config(cfg, globals_mode="points_2000", output_dir=None):
    """Run the exact r3_bn_m32t protocol using an explicitly selected global-feature source."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = build_data(cfg, "spectral")
    for name, loader in (("train", tr), ("val", va), ("test", te)):
        if loader.x.shape[1] != MAX_POINTS:
            raise ValueError(f"{name} has {loader.x.shape[1]} points; expected {MAX_POINTS}")

    if globals_mode == "points_2000":
        globals_ = {name: point_globals(loader.x) for name, loader in
                    (("tr", tr), ("va", va), ("te", te))}
    elif globals_mode == "complete_input":
        data_dir = Path(cfg.paths.data_dir)
        classes = sorted({d.name for split in ("train", "val", "test")
                          for d in (data_dir / split).iterdir() if d.is_dir()})
        label_to_idx = {name: i for i, name in enumerate(classes)}
        raw = complete_input_statistics_for_splits(
            data_dir, label_to_idx, int(cfg.machine.num_workers)
        )
        expected = (tr.y.cpu(), va.y.cpu(), te.y.cpu())
        actual = tuple(raw[split][1] for split in ("train", "val", "test"))
        if any(not torch.equal(found, wanted) for found, wanted in zip(actual, expected)):
            raise RuntimeError("Raw NPZ scalar order does not match the 2,000-point dataset order")
        globals_ = {name: raw[split][0] for name, split in
                    (("tr", "train"), ("va", "val"), ("te", "test"))}
    else:
        raise ValueError(f"Unknown globals_mode={globals_mode!r}")
    mu = globals_["tr"].mean(0)
    sd = globals_["tr"].std(0).clamp_min(1e-6)
    data = dict(
        nc=int(cfg.train.out_channels),
        tr_x=tr.x,
        va_x=va.x,
        te_x=te.x,
        tr_y=tr.y.cpu(),
        va_y=va.y.cpu(),
        te_y=te.y.cpu(),
        g_tr=(globals_["tr"] - mu) / sd,
        g_va=(globals_["va"] - mu) / sd,
        g_te=(globals_["te"] - mu) / sd,
    )
    model = cfg.model
    train = cfg.train
    net = {
        "n_channels": model.n_channels,
        "n_mix": model.n_mix,
        "pow_mode": model.pow_mode,
        "gram": model.gram,
        "triples": model.triples,
        "bispec_lmax": model.bispec_lmax,
        "bispec_mode": model.bispec_mode,
        "norm": model.norm,
        "in_bn": model.in_bn,
        "widths": tuple(model.widths),
        "head_dropout": model.head_dropout,
        "feat_dropout": model.feat_dropout,
    }
    training = {
        "lr": model.learning_rate,
        "wd": model.weight_decay,
        "epochs": train.epochs,
        "bs": cfg.machine.batch_size,
        "label_smoothing": model.label_smoothing,
        "glob": model.use_globals,
        "glob_noise": model.glob_noise,
    }
    result = run_job(
        "spectral",
        net,
        training,
        int(cfg.random_seed),
        data,
        dev,
        n_shells=model.n_shells,
        lmax=model.lmax,
        r_max=model.r_max,
        point_dropout=model.point_dropout,
        density_noise=model.density_noise,
        clean_probability=model.clean_probability,
    )
    print(
        f"spectral seed {cfg.random_seed}: ep {result['epoch'] + 1:2d} | "
        f"val acc {result['val_acc']:.4f} loss {result['val_loss']:.4f} | "
        f"test acc {result['test_acc']:.4f} rec {result['test_rec']:.4f} | "
        f"{result['params'] / 1e6:.2f}M params, d_in {result['d_in']}, "
        f"{result['s_per_epoch']:.1f}s/ep",
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()