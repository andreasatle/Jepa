"""Snapshot the live training checkpoint once per epoch, without touching train.py.

train.py overwrites a single jepa_ckpt.pt at every epoch end, so there is
nothing for eval_probe.py / sweep_probe.py to sweep over. Run this alongside a
training job that predates the `archive_every` change (or any run where you'd
rather not restart):

    uv run python i_jepa/archive_ckpts.py            # poll until Ctrl-C
    uv run python i_jepa/archive_ckpts.py --once     # snapshot current, then exit
    uv run python i_jepa/archive_ckpts.py --interval 300

It writes ENCODER-ONLY snapshots (+ epoch/step) into i_jepa/ckpt_archive/,
~90 MB each vs the ~290 MB full checkpoint -- the same key layout train.py's
archive_encoder() and eval_probe.build_encoder() use. Epochs already present in
the archive are skipped, so it is safe to stop and restart.
"""

import argparse
import os
import pickle
import time
import zipfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(_HERE, "jepa_ckpt.pt")
ARCHIVE = os.path.join(_HERE, "ckpt_archive")

# torch.load hitting the file mid-write shows up as one of these
_PARTIAL = (EOFError, RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile)


def _seen_epochs():
    if not os.path.isdir(ARCHIVE):
        return set()
    out = set()
    for f in os.listdir(ARCHIVE):
        if f.startswith("e") and "_s" in f:
            try:
                out.add(int(f[1:f.index("_s")]))
            except ValueError:
                pass
    return out


def snapshot(seen):
    if not os.path.exists(LIVE):
        return
    try:
        ck = torch.load(LIVE, map_location="cpu")
    except _PARTIAL:
        return                                   # mid-write; try again next tick
    epoch, step = int(ck.get("epoch", -1)), int(ck.get("step", -1))
    if epoch < 0 or epoch in seen:
        return
    os.makedirs(ARCHIVE, exist_ok=True)
    out = os.path.join(ARCHIVE, f"e{epoch:03d}_s{step}.pt")
    torch.save({"version": ck.get("version", "?"), "encoder": ck["encoder"],
                "epoch": epoch, "step": step}, out)
    seen.add(epoch)
    print(f"[archive] epoch {epoch} step {step} -> {out}  "
          f"({os.path.getsize(out) / 1e6:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=120.0,
                    help="seconds between polls (default 120)")
    ap.add_argument("--once", action="store_true", help="snapshot current state and exit")
    args = ap.parse_args()

    seen = _seen_epochs()
    if seen:
        print(f"[archive] already have epochs {sorted(seen)}", flush=True)
    snapshot(seen)
    if args.once:
        return

    print(f"[archive] watching {LIVE} every {args.interval:.0f}s -- Ctrl-C to stop",
          flush=True)
    try:
        while True:
            time.sleep(args.interval)
            snapshot(seen)
    except KeyboardInterrupt:
        print("\n[archive] stopped", flush=True)


if __name__ == "__main__":
    main()
