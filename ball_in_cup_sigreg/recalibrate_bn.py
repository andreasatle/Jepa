"""
Precise BatchNorm (Wu & Johnson, "Rethinking 'Batch' in BatchNorm", 2021): after training, recompute the
projector's BatchNorm running statistics as a plain average over many training batches, with the weights
frozen, and write them back into jepa_checkpoint_strided.pt.

Why (run 2, 2026-11-07): the pre-BN features ended with a large common offset (mean |1.47|) and a tiny
spread (std 0.008), so the running mean's usual EMA noise of a fraction of a percent moved eval-mode z by
about one standard deviation. The planner runs E in eval mode, so P saw shifted inputs: one-step
prediction loss 0.97 in train mode, 24.4 in eval mode, and 29/100 solved. Recalibrated: 0.94.
"""

import torch
from torch.utils.data import DataLoader

import jepa_train as jt
import jepa_train_strided as js

N_BATCHES = 200


def main():
    ckpt = torch.load(js.CHECKPOINT_PATH, map_location="cpu")
    E = jt.Encoder(3, ckpt.get("latent_dim", jt.LATENT_DIM))
    E.load_state_dict(ckpt["E"])
    bn = E.projector[1]
    before = bn.running_mean.clone()
    bn.reset_running_stats()
    bn.momentum = None                      # cumulative average instead of an EMA
    E.train()
    torch.manual_seed(0)
    ds = js.StridedJepaTransitionDataset(js.DATA_DIR, jt.K_CONTEXT, js.UNROLL_STEPS, js.PREDICTOR_STRIDE)
    loader = DataLoader(ds, batch_size=js.BATCH_SIZE, shuffle=True, drop_last=True)
    with torch.no_grad():
        for i, (c, _, t) in enumerate(loader):
            if i == N_BATCHES:
                break
            n, k, ch, h, w = c.shape
            E(torch.cat([c.reshape(-1, ch, h, w), t.reshape(-1, ch, h, w)]))
    shift = ((bn.running_mean - before) / bn.running_var.sqrt()).abs().mean().item()
    print(f"recalibrated BatchNorm over {i} batches; running mean moved {shift:.2f} std per dim on average")
    bn.momentum = 0.1
    ckpt["E"] = E.state_dict()
    torch.save(ckpt, js.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
