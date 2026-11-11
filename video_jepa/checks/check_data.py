"""Checks for data.py (step 1 of PLAN.md): clips, labels and the split.

    uv run python checks/check_data.py

1. Frames match the state: the blue pusher's centroid in each frame lines up with the state's pusher
   position at lag 0, not at the step before or after (an off-by-one would make every velocity label wrong).
2. Velocities: one-step differences of the pusher's position track the simulator's recorded velocity,
   and the two-step labels do too, a little more loosely.
3. Clips: frames STRIDE steps apart, all inside one episode.
4. Split: no held-out episode in the training clips, and the other way round.
5. Contact: the T moves far more often when the pusher touches it than when it doesn't.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import FRAMES, HZ, STRIDE, TARGETS, Clips, PushTVideo


def pusher_centroids(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (col, row) of RoyalBlue pixels in each 64 px frame, and whether enough were found."""
    f = frames.astype(int)
    mask = (f[..., 2] > 150) & (f[..., 0] < 130) & (f[..., 2] - f[..., 1] > 60)
    rows, cols = np.mgrid[0:64, 0:64]
    n = mask.sum((1, 2))
    cx = (mask * cols).sum((1, 2)) / np.maximum(n, 1) + .5
    cy = (mask * rows).sum((1, 2)) / np.maximum(n, 1) + .5
    return np.stack([cx, cy], 1), n >= 4


def check_alignment(v: PushTVideo, rng) -> None:
    ends = v.clip_ends("train")
    rows = np.sort(rng.choice(ends, 3000, replace=False))
    c, ok = pusher_centroids(np.array(v.frames[rows]))
    print(f"pusher visible in {ok.mean():.0%} of {len(rows)} sampled frames")
    errs = {}
    for lag in (-1, 0, 1):
        p = v.state[rows[ok] + lag, 0:2]
        A = np.column_stack([p, np.ones(len(p))])                      # affine fit, pixel = A @ coef
        coef, *_ = np.linalg.lstsq(A, c[ok], rcond=None)
        errs[lag] = np.sqrt(((A @ coef - c[ok]) ** 2).sum(1).mean())
        if lag == 0:
            scale = coef[0, 0], coef[1, 1]
    print("pusher centroid vs state, rms px by lag: " + ", ".join(f"{k:+d}: {e:.2f}" for k, e in errs.items()),
          f"(scale {scale[0]:.4f}, {scale[1]:.4f}; 64/512 = {64 / 512:.4f})")
    assert min(errs, key=errs.get) == 0, "frames and state are offset"
    assert errs[0] < 1.5, "the pusher is not where the state says"


def check_velocity(v: PushTVideo, rng) -> None:
    rows = np.sort(rng.choice(v.clip_ends("train"), 50_000, replace=False))
    rec = v.state[rows, 5:7]
    one = (v.state[rows, 0:2] - v.state[rows - 1, 0:2]) * HZ
    lab = v.labels(rows)[:, [TARGETS.index("pusher_vx"), TARGETS.index("pusher_vy")]]
    for name, est in (("1-step difference", one), (f"{STRIDE}-step label", lab)):
        r = [np.corrcoef(est[:, k], rec[:, k])[0, 1] for k in (0, 1)]
        s = (est * rec).sum() / (est * est).sum()
        print(f"recorded pusher velocity vs {name}: corr x {r[0]:.3f}, y {r[1]:.3f}; recorded = {s:.2f} x estimate")
        assert min(r) > (0.97 if name.startswith("1") else 0.85), f"{name} does not track the recorded velocity"


def check_clips_and_split(v: PushTVideo) -> None:
    tr, ho = set(v.episodes["train"].tolist()), set(v.episodes["held_out"].tolist())
    assert not tr & ho, "an episode is in both splits"
    for split, eps in (("train", tr), ("held_out", ho)):
        ends = v.clip_ends(split)
        first = ends - (FRAMES - 1) * STRIDE
        assert (v.episode_of(first) == v.episode_of(ends)).all(), "a clip crosses an episode boundary"
        assert set(np.unique(v.episode_of(ends)).tolist()) <= eps, f"{split} clips come from the other split"
        print(f"{split}: {len(eps)} episodes, {len(ends):,} possible clips")
    rows = PushTVideo.clip_rows(int(v.clip_ends("train")[0]))
    assert len(rows) == FRAMES and (np.diff(rows) == STRIDE).all()


def check_contact(v: PushTVideo, rng) -> None:
    rows = np.sort(rng.choice(v.clip_ends("train"), 50_000, replace=False))
    lab = v.labels(rows)
    contact = lab[:, TARGETS.index("contact")] > .5
    t_speed = np.hypot(lab[:, TARGETS.index("t_vx")], lab[:, TARGETS.index("t_vy")])
    omega = np.abs(lab[:, TARGETS.index("t_omega")])
    moving = (t_speed > 2) | (omega > .05)
    p_c, p_n = moving[contact].mean(), moving[~contact].mean()
    print(f"contact in {contact.mean():.0%} of frames; T moving: {p_c:.0%} with contact, {p_n:.0%} without")
    assert .05 < contact.mean() < .95, "contact is almost constant"
    assert p_c > 3 * p_n, "the T should mostly move when touched"


def main():
    rng = np.random.default_rng(0)
    v = PushTVideo()
    check_clips_and_split(v)
    check_alignment(v, rng)
    check_velocity(v, rng)
    check_contact(v, rng)
    ds = Clips(v, "held_out", n=2000, seed=2)
    x, y = ds[0]
    assert tuple(x.shape) == (FRAMES, 3, 64, 64) and x.dtype.is_floating_point is False
    assert np.isfinite(ds.targets).all()
    print(f"held-out probe set: {len(ds)} clips, frames {tuple(x.shape)} {x.dtype}")
    print("label    " + " ".join(f"{t:>9}" for t in TARGETS))
    for name, f in (("mean", np.mean), ("std", np.std)):
        print(f"{name:<8} " + " ".join(f"{val:9.2f}" for val in f(ds.targets, 0)))
    print("OK")


if __name__ == "__main__":
    main()
