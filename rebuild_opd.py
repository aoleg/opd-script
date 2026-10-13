"""Rebuild kroma-v0.3.1-turbo-opd from Krea 2 Raw, Krea 2 Turbo and the Kroma teacher.

Recipe, per tensor:
  text side (txtfusion.*, txtmlp.*):  OUT = TEACH
  everything else:                    OUT = TURBO + trim(TEACH - RAW)
trim zeroes every delta element whose magnitude is not above the 0.9 quantile of
|delta| (torch.quantile, linear interpolation), as the published OPD does. Output dtypes follow RAW. Tensors are read and written one at a time.

Optional --compare OPD checks the result against an existing OPD file, element by element.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
COPY_PREFIXES = ("txtfusion.", "txtmlp.")


class Reader:
    """Plain seek + read per tensor (no mmap: a mapped 25 GB file inflates the working set)."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        self.meta = header.pop("__metadata__", None) or {}
        self.infos = header
        self.start = 8 + n
        self.path = path
        self.f = open(path, "rb", buffering=0)
        bad = sorted({v["dtype"] for v in header.values()} - set(DTYPES))
        if bad:
            sys.exit(f"{path}: dtypes {bad} are not supported; a bf16 or fp32 file is needed (no fp8 / int8)")

    def get(self, name: str, device) -> torch.Tensor:
        info = self.infos[name]
        a, b = info["data_offsets"]
        buf = bytearray(b - a)
        self.f.seek(self.start + a)
        view, got = memoryview(buf), 0
        while got < len(buf):
            k = self.f.readinto(view[got:])
            if not k:
                sys.exit(f"{self.path}: short read on {name}")
            got += k
        t = torch.frombuffer(buf, dtype=torch.uint8).view(DTYPES[info["dtype"]]).reshape(info["shape"])
        return t.to(device, torch.float32)


def trim(delta: torch.Tensor, density: float, mode: str = "quantile") -> torch.Tensor:
    """Zero all but the largest-magnitude entries of a delta.

    quantile (default, matches the published OPD): tau = the (1 - density) quantile of |delta|
    with linear interpolation, as torch.quantile computes it; keep |delta| > tau. Ties at tau
    are dropped, which matters for bf16 deltas.
    topk: keep the exact top `density` share by magnitude, ties at the threshold kept.
    """
    if density >= 1.0:
        return delta
    a = delta.abs().reshape(-1)
    n = a.numel()
    if mode == "topk":
        k = max(1, int(round(n * density)))
        if k >= n:
            return delta
        tau = torch.kthvalue(a, n - k + 1).values
        return torch.where(delta.abs() >= tau, delta, torch.zeros_like(delta))
    # torch.quantile refuses inputs above 16M elements, so the interpolation is done here
    # from the two order statistics around position q (n - 1)
    pos = (1.0 - density) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    v_lo = torch.kthvalue(a, lo + 1).values
    v_hi = torch.kthvalue(a, hi + 1).values if hi != lo else v_lo
    tau = v_lo + (pos - lo) * (v_hi - v_lo)
    return torch.where(delta.abs() > tau, delta, torch.zeros_like(delta))


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Krea 2 Raw, bf16/fp32")
    ap.add_argument("--turbo", required=True, help="Krea 2 Turbo, bf16/fp32")
    ap.add_argument("--teach", required=True, help="kroma-sensei-booru-e6-teacher-velocity-v0.3, bf16/fp32")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--density", type=float, default=0.1, help="share of the delta's elements kept (default 0.1 = trim 0.9)")
    ap.add_argument("--threshold", default="quantile", choices=["quantile", "topk"],
                    help="quantile: keep |delta| > the quantile, ties dropped (matches OPD); topk: exact top share, ties kept")
    ap.add_argument("--copy-prefixes", default=",".join(COPY_PREFIXES),
                    help="tensors taken from TEACH as they are (default txtfusion.,txtmlp.)")
    ap.add_argument("--compare", default=None, metavar="OPD", help="check the output against this OPD file afterwards")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    copy = tuple(p for p in args.copy_prefixes.split(",") if p)
    R = {k: Reader(getattr(args, k)) for k in ("raw", "turbo", "teach")}
    names = sorted(R["raw"].infos, key=lambda k: R["raw"].infos[k]["data_offsets"][0])
    for k in ("turbo", "teach"):
        missing = [n for n in names if n not in R[k].infos]
        wrong = [n for n in names if n in R[k].infos and R[k].infos[n]["shape"] != R["raw"].infos[n]["shape"]]
        if missing or wrong:
            sys.exit(f"{k}: {len(missing)} tensors missing, {len(wrong)} with another shape, e.g. {(missing + wrong)[:3]}")

    # the header is known before any data: same names, shapes and dtypes as RAW
    header, offset = {}, 0
    for n in names:
        info = R["raw"].infos[n]
        size = SIZES[info["dtype"]] * int(np.prod(info["shape"], dtype=np.int64))
        header[n] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {
        "recipe": "TURBO + trim(TEACH - RAW); copy_prefixes from TEACH",
        "base": os.path.basename(args.raw), "target": os.path.basename(args.turbo),
        "finetune": os.path.basename(args.teach), "trim": f"{1 - args.density:g}",
        "copy_prefixes": ",".join(copy), "scale": "1.0", "threshold": args.threshold}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)

    t0 = time.time()
    tmp = args.output + ".part"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(raw_header)))
        out.write(raw_header)
        for i, n in enumerate(names):
            target_dtype = DTYPES[R["raw"].infos[n]["dtype"]]
            if n.startswith(copy):
                result = R["teach"].get(n, dev)
            else:
                delta = R["teach"].get(n, dev) - R["raw"].get(n, dev)
                result = R["turbo"].get(n, dev) + trim(delta, args.density, args.threshold)
                del delta
            out.write(tensor_bytes(result.to(target_dtype)))
            del result
            if (i + 1) % 50 == 0 or i + 1 == len(names):
                print(f"[{i + 1}/{len(names)}] {time.time() - t0:.0f} s", flush=True)
    os.replace(tmp, args.output)
    print(f"wrote {args.output}")

    if args.compare:
        compare(args.output, args.compare, R["turbo"], copy, dev)


def compare(built: str, opd: str, turbo: Reader, copy: tuple, dev):
    """Per prefix: share of bit-identical elements, and ||built - OPD|| / ||OPD - TURBO||."""
    A, B = Reader(built), Reader(opd)
    stats = {}
    for n in A.infos:
        if n not in B.infos:
            print(f"not in OPD: {n}")
            continue
        a, b = A.get(n, dev), B.get(n, dev)
        group = next((p for p in copy if n.startswith(p)), n.split(".")[0] + ".")
        s = stats.setdefault(group, [0, 0, 0.0, 0.0, 0])
        s[0] += int((a == b).sum())
        s[1] += a.numel()
        s[2] += float(((a - b).double() ** 2).sum())
        s[3] += float(((b - turbo.get(n, dev)).double() ** 2).sum())
        s[4] += int(A.infos[n]["dtype"] != B.infos[n]["dtype"])
    print(f"\n{'prefix':12s} {'identical':>10s} {'rel. error':>11s} {'dtype diff':>10s}")
    tot = [0, 0, 0.0, 0.0]
    for g, (same, cnt, e2, d2, dt) in sorted(stats.items()):
        for j, v in enumerate((same, cnt, e2, d2)):
            tot[j] += v
        rel = (e2 ** 0.5) / (d2 ** 0.5) if d2 else float("nan")
        print(f"{g:12s} {same / cnt:10.5f} {rel:11.2e} {dt:10d}")
    print(f"{'all':12s} {tot[0] / tot[1]:10.5f} {(tot[2] ** 0.5) / (tot[3] ** 0.5):11.2e}")


if __name__ == "__main__":
    main()
