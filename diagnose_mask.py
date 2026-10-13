"""Compare the trim mask of the published OPD with a plain top-k mask, tensor by tensor.

For each tensor: Δ = TEACH − RAW; the OPD mask is where OPD differs from TURBO; the rebuilt
mask is the exact top-k of |Δ|. Reports kept shares, the magnitude ranges of both masks,
and whether the OPD mask is a clean magnitude threshold (all kept |Δ| above all dropped |Δ|).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild_opd import Reader  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("raw", "turbo", "teach", "opd", "built"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--tensors", nargs="+", required=True)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = {k: Reader(getattr(args, k)) for k in ("raw", "turbo", "teach", "opd", "built")}
    print(f"{'tensor':30s} {'dtype':5s} {'numel':>10s} {'OPD kept':>9s} {'top-k kept':>10s} {'differ':>8s} "
          f"{'τ exact':>10s} {'OPD min kept':>12s} {'OPD max drop':>12s} {'clean':>5s}")
    for n in args.tensors:
        t = {k: R[k].get(n, dev) for k in R}
        d = t["teach"] - t["raw"]
        opd_mask = t["opd"] != t["turbo"]
        built_mask = t["built"] != t["turbo"]
        a = d.abs()
        k = round(0.1 * d.numel())
        tau = torch.kthvalue(a.reshape(-1), d.numel() - k + 1).values.item()
        kept_min = a[opd_mask].min().item() if opd_mask.any() else float("nan")
        drop_max = a[~opd_mask].max().item()
        print(f"{n:30s} {R['raw'].infos[n]['dtype']:5s} {d.numel():10d} {opd_mask.float().mean().item():9.5f} "
              f"{built_mask.float().mean().item():10.5f} {(opd_mask != built_mask).float().mean().item():8.5f} "
              f"{tau:10.3e} {kept_min:12.3e} {drop_max:12.3e} {str(kept_min > drop_max):>5s}")
        del t, d, a


if __name__ == "__main__":
    main()
