"""Does the message-passing baseline fit on this GPU, and how fast is it?

Runs a few forward+backward steps on random point clouds (no dataset needed) and prints
peak GPU memory and estimated epoch time for several batch sizes.

Usage:
    uv run -m src.scripts.probe_graph
    uv run -m src.scripts.probe_graph --mul 32 --n_points 256 --k 12 --batches 8 16 32
"""
import argparse
import time

import torch

from src.models.e3nn.graph_model import E3NNGraphModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--n_points", type=int, default=256)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--mul", type=int, default=32)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--train_size", type=int, default=41105)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = E3NNGraphModel(
        num_classes=args.num_classes, n_points=args.n_points, k=args.k,
        mul=args.mul, lmax=args.lmax, n_layers=args.layers,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M | n_points={args.n_points} k={args.k} mul={args.mul} "
          f"lmax={args.lmax} layers={args.layers} | device={dev}")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free/total: {free / 1e9:.1f} / {total / 1e9:.1f} GB")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for B in args.batches:
        x = y = None
        try:
            x = torch.rand(B, 2000, 4, device=dev)
            x[:, :, 3] = torch.rand(B, 2000, device=dev).sort(dim=1, descending=True).values
            x[:, 1500:, 3] = 0  # some padding
            y = torch.randint(0, args.num_classes, (B,), device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for s in range(args.steps):
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
            step_t = sum(times[1:]) / max(1, len(times) - 1)  # skip warm-up step
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
            epoch_s = step_t * args.train_size / B
            print(f"batch {B:4d}: peak {peak:6.2f} GB | {step_t * 1000:7.1f} ms/step | "
                  f"~{epoch_s:6.0f} s/epoch (train split)")
        except torch.OutOfMemoryError:
            print(f"batch {B:4d}: OUT OF MEMORY")
            torch.cuda.empty_cache()
            break
        finally:
            del x, y
            if dev.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()