"""
Converts LeWorldModel's Push-T dataset (data/pusht_expert_train.h5: 18,685 expert episodes, 2.3M frames
at 224 px) into what training and evaluation read:

  data/frames64.npy   (N, 64, 64, 3) uint8, every frame resized with cv2.INTER_AREA (evaluate.py resizes
                      the simulator's 224 px renders the same way, so model inputs match)
  data/meta.npz       action (N, 2), state (N, 7), ep_offset, ep_len, and the episode split:
                      5% of episodes held out (seed 0); evaluation start states come only from those.

    uv run python prepare_data.py
"""

import cv2
import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filter the pixels are stored with)
import numpy as np

SRC = "data/pusht_expert_train.h5"
SIZE = 64
CHUNK = 10_000
HELD_OUT = 0.05


def main():
    f = h5py.File(SRC, "r")
    n = f["pixels"].shape[0]
    out = np.lib.format.open_memmap("data/frames64.npy", mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    for i in range(0, n, CHUNK):
        x = f["pixels"][i:i + CHUNK]
        out[i:i + len(x)] = np.stack([cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for im in x])
        if i % 200_000 == 0:
            print(f"{i}/{n}", flush=True)
    out.flush()
    n_ep = len(f["ep_len"])
    perm = np.random.default_rng(0).permutation(n_ep)
    held = np.sort(perm[: round(HELD_OUT * n_ep)])
    train = np.sort(perm[round(HELD_OUT * n_ep):])
    np.savez("data/meta.npz", action=f["action"][:], state=f["state"][:], ep_offset=f["ep_offset"][:],
             ep_len=f["ep_len"][:], train_episodes=train, held_out_episodes=held)
    print(f"done: {n} frames, {len(train)} train / {len(held)} held-out episodes")


if __name__ == "__main__":
    main()
