"""Run eval_probe.py over every archived checkpoint, then redraw the curve.

eval_probe.py already appends one row per checkpoint step to
probe_out/probe_log.csv, and `--plot-log` renders probe accuracy + effective
rank vs pretraining step. This just loops it over i_jepa/ckpt_archive/*.pt
(oldest step first) and passes --plot-log on the final run -- the curve to watch
instead of eyeballing t-SNE.

    uv run python i_jepa/sweep_probe.py                       # kNN + linear, CPU, all of train
    uv run python i_jepa/sweep_probe.py --n-train 20000       # faster: subsample train
    uv run python i_jepa/sweep_probe.py --device mps --tokens # + attentive probe (shares the GPU)
    uv run python i_jepa/sweep_probe.py --live --random       # also probe jepa_ckpt.pt and a random init
    uv run python i_jepa/sweep_probe.py --dry-run

Unknown args (--device, --tokens, --n-train, --knn-k, ...) are forwarded
verbatim to eval_probe.py for every run. Features are cached per checkpoint
under probe_cache/, so a re-run only re-probes; delete probe_cache/ to force
re-extraction.
"""

import argparse
import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(_HERE, "ckpt_archive")
LIVE = os.path.join(_HERE, "jepa_ckpt.pt")
PROBE = os.path.join(_HERE, "eval_probe.py")


def step_of(path):
    """Sort key: the integer after `_s` in e012_s45678.pt (0 if absent)."""
    b = os.path.basename(path)
    if "_s" in b:
        try:
            return int(b[b.index("_s") + 2:].split(".")[0])
        except ValueError:
            pass
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="sweep eval_probe.py over i_jepa/ckpt_archive/*.pt")
    ap.add_argument("--live", action="store_true",
                    help="also probe the current jepa_ckpt.pt (appended last)")
    ap.add_argument("--random", action="store_true",
                    help="also probe a seeded random-init encoder (eval_probe --ckpt none)")
    ap.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    ckpts = sorted(glob.glob(os.path.join(ARCHIVE, "*.pt")), key=step_of)
    targets = (["none"] if args.random else []) + ckpts
    if args.live and os.path.exists(LIVE):
        targets.append(LIVE)

    if not targets:
        sys.exit(
            f"no checkpoints in {ARCHIVE}\n"
            f"  -> run `uv run python i_jepa/archive_ckpts.py` alongside training,\n"
            f"     or pass --live to probe {os.path.basename(LIVE)} on its own.")

    print(f"sweep plan ({len(targets)} run(s), forwarding {extra or 'no extra args'}):")
    for i, t in enumerate(targets):
        print(f"  {i + 1:2d}. {t}" + ("   [+--plot-log]" if i == len(targets) - 1 else ""))
    if args.dry_run:
        return

    for i, t in enumerate(targets):
        cmd = [sys.executable, PROBE, "--ckpt", t, *extra]
        if i == len(targets) - 1:
            cmd.append("--plot-log")
        print(f"\n=== [{i + 1}/{len(targets)}] {t} ===\n$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    print(f"\ndone -> {os.path.join(_HERE, 'probe_out', 'probe_log.png')}")


if __name__ == "__main__":
    main()
