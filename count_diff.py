"""Count differing elements between two safetensors files, per tensor (bit-level, after decoding)."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild_opd import Reader  # noqa: E402

a, b = Reader(sys.argv[1]), Reader(sys.argv[2])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total = diff = 0
for n in a.infos:
    x, y = a.get(n, dev), b.get(n, dev)
    k = int((x != y).sum())
    total += x.numel()
    diff += k
    if k:
        m = (x != y)
        print(f"{n}: {k} of {x.numel()} differ, max |diff| {float((x - y).abs().max()):.3e}, "
              f"|value| there {float(y.abs()[m].mean()):.3e}, dtype {a.infos[n]['dtype']}/{b.infos[n]['dtype']}")
print(f"total: {diff} of {total} elements differ ({diff / total:.2e})")
