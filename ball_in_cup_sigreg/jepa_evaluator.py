"""
World-model evaluator -- given a real starting state and a sequence of
actions, progress the state forward through the trained predictor P.

This is exactly the "P^10" question made concrete and runnable: P doesn't
act on a bare embedding, it acts on the sliding K_CONTEXT-length window
(z_{t-2}, z_{t-1}, z_t), and needs one action per step, so there's no single
operator you can just exponentiate -- what you actually iterate is

    S_a(window) = ( window[1:], P(window, a) )        -- predict, then slide

and an H-step rollout is the composition S_{a_{t+H-1}} o ... o S_{a_t},
applied to a window bootstrapped from K_CONTEXT real frames. E is only ever
called once, up front, to build that initial window -- every step after
that is pure P, no pixels involved.

Two things this is built to serve, both of which are exactly this same
operation -- (initial window, action sequence) -> sequence of predicted
embeddings -- differing only in what's done with the output afterward:

  1. The CEM planner's cost function: roll a BATCH of candidate action
     sequences forward, decode predicted positions (via the linear probe,
     once it's persisted), score each candidate.
  2. The autoregressive drift diagnostic: roll P forward on a single REAL
     recorded action sequence and compare predicted vs. true position, to
     find out empirically how many steps before compounding error makes
     the rollout untrustworthy -- the open question from the "P^10" /
     horizon-length discussion.

Batched throughout (leading dim B) for exactly that reason -- CEM needs to
score many candidates at once, but B=1 works fine for the single-episode
diagnostic case too.

Run as a script for a smoke test against the real checkpoint + one real
episode (no training, no gradient, just shape/sanity checks):
    python3 jepa_evaluator.py
"""

import pathlib

import numpy as np
import torch

import jepa_train as jt

CHECKPOINT_PATH = "jepa_checkpoint.pt"
EVAL_DATA_DIR = pathlib.Path("jepa_eval_data")


def load_world_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    """Loads BOTH E and P from a checkpoint (unlike eval_linear_probe.py's
    load_encoder, which only needs E for probing -- an evaluator needs P
    too, since that's what actually progresses the state).
    Returns (E, P, k_context, latent_dim)."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    k_context = ckpt.get("k_context", jt.K_CONTEXT)
    latent_dim = ckpt.get("latent_dim", jt.LATENT_DIM)

    E = jt.Encoder(3, latent_dim).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()

    P = jt.Predictor(latent_dim, jt.ACTION_DIM, jt.PREDICTOR_HIDDEN, k_context).to(device)
    P.load_state_dict(ckpt["P"])
    P.eval()

    return E, P, k_context, latent_dim


@torch.no_grad()
def bootstrap_window(E, frames, device):
    """frames: (K_CONTEXT, H, W, 3) uint8 -- the K_CONTEXT real frames ending
    at the current real time t (the ONLY place pixels ever enter an
    evaluator run). Returns (1, K_CONTEXT, D) -- batch dim 1, broadcastable
    to whatever batch size a rollout needs."""
    tensors = jt._frames_to_tensor(frames).to(device)   # (k, 3, H, W)
    z = E(tensors)                                        # (k, D)
    return z.unsqueeze(0)                                 # (1, k, D)


@torch.no_grad()
def rollout(P, window, actions):
    """The S_a operator, applied H times -- this IS "progress the state
    given actions".

    window:  (B, K_CONTEXT, D) -- starting window, one per candidate. Pass
             B=1 and it's broadcast automatically if actions has B>1 (the
             common case: many candidate action sequences, one shared real
             starting state).
    actions: (B, H, ACTION_DIM) -- one action sequence per candidate.

    Returns z_preds: (B, H, D) -- the predicted embedding at every one of
    the H steps (z_preds[:, -1] is the final predicted state, matching
    what H calls to S_a would give you by hand).
    """
    B, H, _ = actions.shape
    if window.shape[0] == 1 and B > 1:
        window = window.expand(B, -1, -1).contiguous()
    else:
        assert window.shape[0] == B, (
            f"window batch {window.shape[0]} doesn't match actions batch {B} "
            "and isn't 1 (broadcastable)."
        )

    window = window.clone()
    z_preds = []
    for tau in range(H):
        a = actions[:, tau, :]
        z_next = P(window, a)                             # (B, D)
        window = torch.cat([window[:, 1:, :], z_next.unsqueeze(1)], dim=1)
        z_preds.append(z_next)

    return torch.stack(z_preds, dim=1)                     # (B, H, D)


def _smoke_test():
    """No training, no gradients -- just confirms the evaluator runs
    end-to-end against the real checkpoint and real recorded data, and
    that every shape is what the docstrings above say it is."""
    device = torch.device("cpu")
    E, P, k_context, latent_dim = load_world_model(device=device)
    print(
        f"Loaded {CHECKPOINT_PATH}: k_context={k_context}, latent_dim={latent_dim}"
    )

    paths = sorted(EVAL_DATA_DIR.glob("episode_*.npz"))
    if not paths:
        print(f"No episodes in {EVAL_DATA_DIR}/ -- skipping the data-driven part.")
        return
    data = np.load(paths[0])
    frames = data["frames"]     # (T+1, H, W, 3)
    actions = data["actions"]   # (T, action_dim)
    print(f"Loaded {paths[0].name}: {frames.shape[0]} frames, {actions.shape[0]} actions.")

    # Bootstrap from the first k_context real frames.
    window = bootstrap_window(E, frames[:k_context], device)
    print(f"Initial window shape: {tuple(window.shape)}  (expect (1, {k_context}, {latent_dim}))")

    # Roll forward H=20 real steps using the REAL recorded actions --
    # single sequence, B=1, exactly the shape the drift diagnostic needs.
    H = 20
    action_seq = torch.from_numpy(actions[k_context - 1 : k_context - 1 + H].astype(np.float32))
    action_seq = action_seq.unsqueeze(0)   # (1, H, action_dim)
    z_preds = rollout(P, window, action_seq)
    print(f"z_preds shape: {tuple(z_preds.shape)}  (expect (1, {H}, {latent_dim}))")

    # And the batched case CEM will actually use: B candidate sequences,
    # one shared real starting window, broadcast automatically.
    B = 8
    random_actions = torch.rand(B, H, jt.ACTION_DIM) * 2 - 1   # in [-1, 1]
    z_preds_batched = rollout(P, window, random_actions)
    print(
        f"Batched z_preds shape: {tuple(z_preds_batched.shape)}  "
        f"(expect ({B}, {H}, {latent_dim}))"
    )

    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
