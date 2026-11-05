"""
jepa_train_step.py

Maze JEPA -- Phase 3, step 1: train a ONE-STEP, action-conditioned
predictor P_step(E(cell), action) -> E(next cell) on top of a FROZEN,
already-trained E (loaded from jepa_train.py's checkpoint). See
jepa_model.py's OneStepPredictor docstring for why this is a different
model from Phase 2's P (which deliberately has no action input), and
WORKPLAN.md's Phase 3 section for the full reasoning.

E is frozen (requires_grad=False), not retrained jointly here -- the
question this answers is whether the representation Phase 2 already
found (and picked a geometry for, at LATENT_DIM=8) supports one-step
planning, not "what's the best possible representation for one-step
prediction," which would be a different, easier, confounded question.

Data, THREE sources, the third optional:
  1. maze_data.json -- random-walk rollouts (collect_data.py).
  2. maze_data_directed.json -- goal-directed rollouts, a hand-authored
     approximation of what CEM's candidates look like (collect_directed_data.py).
  3. maze_data_dagger_<geometry>.json -- REAL on-policy rollouts, collected
     by actually running the current planner and recording what it really
     does (collect_dagger_data.py, DAgger-style). Geometry-specific,
     unlike the first two (pure maze physics, independent of geometry).
     OPTIONAL: this file only exists after collect_dagger_data.py has been
     run at least once for this geometry -- before that (e.g. the very
     first training run this project ever does), training proceeds on
     just the first two sources, same as before DAgger existed.
Every episode from every present source is cut into UNROLL_STEPS-length
windows -- a start cell plus UNROLL_STEPS consecutive (action, next_cell)
pairs. Run collect_directed_data.py once before this script if
maze_data_directed.json doesn't exist yet -- unlike the DAgger source,
THIS one is required: this script fails loudly (FileNotFoundError)
rather than silently training on random-walk data alone if it's
missing, since that would silently reproduce the exact
distribution-mismatch bug that file's introduction fixed.

Balanced-ratio batch sampling by source (added after EPISODES_PER_BIAS
500 -> 1500 caused a severe REGRESSION, not an improvement -- see
WORKPLAN.md's "EPISODES_PER_BIAS=1500 regression" section): different
sources' episodes have very different average lengths, so raw episode
count doesn't translate to raw window count, and training batches
sampled uniformly over a straight concatenation of all windows silently
mirror whatever imbalance that produces -- observed once already,
directed windows outnumbering random-walk ones ~3.5:1 at
EPISODES_PER_BIAS=1500 quietly starved the model of general one-step
coverage and collapsed even single-step accuracy. The fix: build every
present source's window pool SEPARATELY (build_transitions called once
per source, never on a concatenated episode list) and sample a FIXED
number of each per batch, regardless of how large any pool is -- an
even split across however many sources (2 or 3) are present. This makes
growing any one pool for broader coverage pure upside -- it can never
again crowd out another source's signal by sheer window count.

The maze's transitions are deterministic given (cell, action) (a wall
bump is still a deterministic no-op -- see env.py), so there's no
multimodality here and no VICReg term is needed: plain supervised
regression against real targets throughout.

Multi-step (unrolled) training -- why, added after the planner failed
0/40 on both geometries: step_drift_diagnostic.py confirmed exposure
bias. P_step was trained one step at a time, always fed the REAL
E(cell) as input (teacher forcing) -- it never once saw its own
predicted output as input during training. But maze_planner.py's CEM
chains P_step AUTOREGRESSIVELY for up to ~45 steps (its own prediction
becomes the next step's input). The diagnostic showed exactly the
signature of that mismatch: teacher-forced error stayed low and flat
regardless of chain length, while autoregressive error grew to 100-1000x
larger by 30-45 steps -- fine one-step accuracy, exploding multi-step
accuracy. Directly precedented by ball_in_cup_hyperbolic/jepa_train.py's
v7 fix (see that module's "Multi-step (unrolled) training" section).

The fix, same idea as that precedent, simpler here since
OneStepPredictor has no context window (just one raw embedding) to
slide: train on UNROLL_STEPS=U successive P_step calls per example.
(U=5 was tried first and confirmed insufficient by a re-run of
step_drift_diagnostic.py -- see WORKPLAN.md's "Phase 3: unroll fix
insufficient alone" section -- U is now 20.)
The first call takes the REAL E(start_cell) as input. Each later call
takes P_step's own PREDICTED embedding from the previous call as input
-- not the real cell's embedding. The TARGET at every one of the U
calls is still the real E(next_cell) from the data -- only the input
degrades across the chain, never the supervision. U=1 reproduces the
original single-step loss exactly. The U per-step losses are averaged
(not summed) into one scalar so the loss scale doesn't shift when U is
changed, and one backward pass runs through the whole chain, so P_step
is trained against its own compounding error instead of only ever
seeing ground-truth input.

Run (in Andreas's terminal, after jepa_train.py has produced
jepa_checkpoint_<geometry>.pt for the geometry being trained here, AND
collect_directed_data.py has produced maze_data_directed.json; run
collect_dagger_data.py first too, any number of times, if DAgger data
should be included):
    uv run python jepa_train_step.py --geometry euclidean
    uv run python jepa_train_step.py --geometry hyperbolic
"""
import argparse
import json
import os

import torch

from env import ACTIONS
from jepa_model import (
    STEP_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    HIDDEN_DIM,
    OneStepPredictor,
    cell_index,
    load_encoder_checkpoint,
    squared_dist,
    to_geometry,
)

DATA_PATH = "maze_data.json"
DIRECTED_DATA_PATH = "maze_data_directed.json"  # collect_directed_data.py -- see
                                                  # module docstring's Data section
DAGGER_DATA_PATH = "maze_data_dagger_{geometry}.json"  # collect_dagger_data.py --
                                                          # OPTIONAL, see Data section
N_STEPS = 10000  # bumped from 2000 -- at 2000 steps the loss was still
                 # dropping steeply (0.81 -> 0.54 in the last 500 steps,
                 # no sign of a plateau), so training wasn't converged
                 # yet. Checking this before assuming undertrained results
                 # were a capacity limit (HIDDEN_DIM).
LR = 0.01
BATCH_SIZE = 256
SEED = 0
UNROLL_STEPS = 8  # U -- see module docstring's "Multi-step (unrolled)
                   # training" section. U=1 reproduces the pre-fix
                   # single-step loss exactly. Was 20 (deliberate middle
                   # point for the size-16 maze, diameter 46, deployment
                   # horizon ~45 -- see WORKPLAN.md's "Bumped from 5 -> 20"
                   # section for that history). Dropped to 8 for the
                   # size-10 maze experiment (diameter 27, MIN_TRIAL_DISTANCE=5):
                   # collect_directed_data.py requires MIN_DISTANCE ==
                   # UNROLL_STEPS (an episode needs to be at least this long
                   # to yield one training window), so at U=20 directed data
                   # only covered the rare near-diameter sliver (3.7% of
                   # pairs) while 84.2% of real eval trials (bfs_distance
                   # in [5,20)) had ZERO goal-directed training coverage --
                   # a confound, not a fair test of the horizon-length
                   # hypothesis. U=8 sits comfortably above
                   # MIN_TRIAL_DISTANCE (5) and well under the diameter (27),
                   # so directed data now covers the range real trials
                   # actually use. Revert to 20 before retraining against
                   # the size-16 maze -- this value is maze-size-specific,
                   # not a universal improvement.

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def build_transitions(episodes, unroll_steps):
    """Every valid UNROLL_STEPS-length window in every collected rollout:
    a start cell, its next `unroll_steps` actions, and the `unroll_steps`
    real cells those actions actually lead to (one per unroll position)."""
    starts, action_seqs, target_seqs = [], [], []
    for ep in episodes:
        states = ep["states"]
        acts = ep["actions"]
        for i in range(len(acts) - unroll_steps + 1):
            starts.append(tuple(states[i]))
            action_seqs.append([ACTION_INDEX[a] for a in acts[i:i + unroll_steps]])
            target_seqs.append([tuple(c) for c in states[i + 1:i + 1 + unroll_steps]])
    return starts, action_seqs, target_seqs


def _to_tensors(starts, action_seqs, target_seqs, size):
    start_idx = torch.tensor([cell_index(s, size) for s in starts])
    action_idx = torch.tensor(action_seqs)  # (N, U)
    target_idx = torch.tensor(
        [[cell_index(c, size) for c in seq] for seq in target_seqs]
    )  # (N, U)
    return start_idx, action_idx, target_idx


def _load_pool(path, size, label):
    with open(path) as f:
        data = json.load(f)
    starts, action_seqs, target_seqs = build_transitions(data["episodes"], UNROLL_STEPS)
    idx_tensors = _to_tensors(starts, action_seqs, target_seqs, size)
    print(f"  {label}: {len(data['episodes'])} episodes -> {len(starts)} windows")
    return idx_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    parser.add_argument("--encoder-checkpoint", default=None,
                         help="Path template for the frozen E checkpoint to load (must "
                              "contain {geometry}). Defaults to the standard jepa_train.py "
                              "output. Pass a different template (e.g. "
                              "jepa_checkpoint_treeloss_{geometry}.pt) to train P_step "
                              "against a differently-trained E -- such as "
                              "jepa_train_treeloss.py's output -- without touching the "
                              "standard checkpoint.")
    parser.add_argument("--step-checkpoint-out", default=None,
                         help="Path template for where to save the trained P_step "
                              "checkpoint (must contain {geometry}). Defaults to the "
                              "standard STEP_CHECKPOINT_PATH. Use a different template "
                              "whenever --encoder-checkpoint points at a non-standard E, "
                              "so this doesn't overwrite the standard P_step checkpoint.")
    parser.add_argument("--use-dagger", action="store_true",
                         help="Include a DAgger data pool in training. OFF by default: "
                              "DAgger's whole value is matching the CURRENT policy's "
                              "visited-state distribution -- rounds collected against a "
                              "DIFFERENT E/P_step pair aren't genuinely on-policy for this "
                              "one, just historical data that happens to still be valid "
                              "maze transitions. Only turn on once shown to help THIS pair, "
                              "not assumed by default because it helped before. Combine "
                              "with --dagger-path when using a non-standard E (see "
                              "collect_dagger_data.py's --output-path).")
    parser.add_argument("--dagger-path", default=None,
                         help="Override the DAgger data file path template (must contain "
                              "{geometry}). Defaults to maze_data_dagger_{geometry}.json. "
                              "Use a different template whenever --encoder-checkpoint points "
                              "at a non-standard E, matching whatever --output-path "
                              "collect_dagger_data.py was given for that E.")
    parser.add_argument("--residual", action="store_true",
                         help="OneStepPredictor predicts start_raw + a small learned "
                              "delta, instead of the whole next raw vector from "
                              "scratch. Off by default (preserves original behavior). "
                              "See jepa_model.py's OneStepPredictor docstring.")
    parser.add_argument("--data-path", default=None,
                         help="override DATA_PATH (random-walk episodes) -- use a "
                              "distinct file when GRID_SIZE differs from the "
                              "standard maze, since episode cell indices are tied "
                              "to a specific maze/GRID_SIZE.")
    parser.add_argument("--directed-data-path", default=None,
                         help="override DIRECTED_DATA_PATH -- same reasoning as "
                              "--data-path.")
    args = parser.parse_args()
    geometry = args.geometry
    encoder_checkpoint_path = args.encoder_checkpoint or CHECKPOINT_PATH
    step_checkpoint_out_path = args.step_checkpoint_out or STEP_CHECKPOINT_PATH

    torch.manual_seed(SEED)

    E, e_ckpt = load_encoder_checkpoint(geometry, path=encoder_checkpoint_path)
    size, dim = e_ckpt["size"], e_ckpt["dim"]
    print(f"loaded frozen E (dim={dim}) from {encoder_checkpoint_path.format(geometry=geometry)}")

    # Every present source is built into its own SEPARATE window pool (never
    # concatenated before build_transitions) so each can be sampled from
    # independently below -- see module docstring's "Balanced-ratio batch
    # sampling by source" section.
    data_path = args.data_path or DATA_PATH
    directed_data_path = args.directed_data_path or DIRECTED_DATA_PATH

    pools = []  # list of (label, (start_idx, action_idx, target_idx))
    pools.append(("random-walk", _load_pool(data_path, size, f"random-walk ({data_path})")))
    pools.append(("directed", _load_pool(directed_data_path, size, f"directed ({directed_data_path})")))

    dagger_path = (args.dagger_path or DAGGER_DATA_PATH).format(geometry=geometry)
    dagger_round_count = 0  # 0 means DAgger data isn't in this training run at all
                             # (either --use-dagger wasn't passed, or no file exists) --
                             # used to tag the checkpoint below
    if not args.use_dagger:
        print(f"  dagger: skipped (pass --use-dagger to include {dagger_path} -- "
              f"off by default, see --use-dagger's help)")
    elif os.path.exists(dagger_path):
        with open(dagger_path) as f:
            dagger_round_count = len(json.load(f)["rounds"])
        pools.append(("dagger", _load_pool(dagger_path, size, f"dagger ({dagger_path})")))
    else:
        print(f"  dagger: --use-dagger passed but {dagger_path} not found -- "
              f"training without it (run collect_dagger_data.py first)")

    n_sources = len(pools)
    per_source_batch = BATCH_SIZE // n_sources
    batch_sizes = [per_source_batch] * n_sources
    batch_sizes[-1] += BATCH_SIZE - per_source_batch * n_sources  # remainder to the last pool
    print(f"sampling {batch_sizes} per training batch from {[label for label, _ in pools]} "
          f"respectively, regardless of each pool's size")

    P_step = OneStepPredictor(num_actions=len(ACTIONS), dim=dim, residual=args.residual)
    opt = torch.optim.Adam(P_step.parameters(), lr=LR)

    pool_sizes = [len(idx_tensors[0]) for _, idx_tensors in pools]

    for step in range(N_STEPS):
        start_parts, action_parts, target_parts = [], [], []
        for (label, (start_idx, action_idx, target_idx)), n, b in zip(pools, pool_sizes, batch_sizes):
            batch = torch.randint(0, n, (b,))
            start_parts.append(start_idx[batch])
            action_parts.append(action_idx[batch])
            target_parts.append(target_idx[batch])
        start_idx_b = torch.cat(start_parts)
        action_idx_b = torch.cat(action_parts)  # (B, U)
        target_idx_b = torch.cat(target_parts)  # (B, U)

        with torch.no_grad():
            raw = E(start_idx_b)  # (B, dim) -- real starting embedding
            target_raw = E(target_idx_b.reshape(-1)).reshape(
                BATCH_SIZE, UNROLL_STEPS, -1
            )  # (B, U, dim) -- real target embeddings, one per unroll position

        step_losses = []
        for t in range(UNROLL_STEPS):
            pred_raw = P_step(raw, action_idx_b[:, t])
            pred_mapped = to_geometry(pred_raw, geometry)
            target_mapped = to_geometry(target_raw[:, t], geometry)
            step_losses.append(squared_dist(pred_mapped, target_mapped, geometry).mean())
            raw = pred_raw  # exposure-bias fix: feed the PREDICTION forward
                             # as next step's input -- never the real target
        loss = torch.stack(step_losses).mean()

        opt.zero_grad()
        loss.backward()
        # Gradient clipping -- added after DAgger round 2 produced a real
        # exploding-gradient crash: loss jumped ~3 -> 166 around step 8500
        # and never recovered, then the corrupted checkpoint produced NaN
        # costs in cem_plan (best_actions stayed None the whole search).
        # Plausible trigger: hyperbolic.py's own self-check docstring notes
        # torch.acosh has an infinite derivative at arg==1 (predicted ==
        # target) -- one DAgger example landing near-exactly on its target
        # mid-unroll is enough to spike the gradient through the whole
        # 20-step unrolled chain. max_norm=1.0 is a standard, untuned
        # starting point, not derived from this project's own numbers.
        torch.nn.utils.clip_grad_norm_(P_step.parameters(), max_norm=1.0)
        opt.step()
        if step % 500 == 0 or step == N_STEPS - 1:
            print(f"  [{geometry}] step {step}: loss={loss.item():.6f}")

    checkpoint_path = step_checkpoint_out_path.format(geometry=geometry)
    ckpt_dict = {
        "P_step": P_step.state_dict(),
        "geometry": geometry,
        "size": size,
        "dim": dim,
        "num_actions": len(ACTIONS),
        "unroll_steps": UNROLL_STEPS,
        "dagger_round_count": dagger_round_count,
        "hidden": HIDDEN_DIM,
        "residual": args.residual,
    }
    torch.save(ckpt_dict, checkpoint_path)
    print(f"checkpoint saved to {checkpoint_path}")

    # ALSO keep a round-tagged copy (e.g. jepa_step_checkpoint_hyperbolic_dagger3.pt)
    # so past DAgger rounds can be compared later (under --replan or otherwise)
    # without needing to re-collect anything -- checkpoint_path alone gets
    # overwritten every run, which is why rounds 1-8's checkpoints weren't kept
    # separately earlier in this project. dagger_round_count=0 still gets tagged
    # (the pre-DAgger baseline), for the same reason.
    round_tagged_path = checkpoint_path.replace(".pt", f"_dagger{dagger_round_count}.pt")
    torch.save(ckpt_dict, round_tagged_path)
    print(f"round-tagged checkpoint also saved to {round_tagged_path}")


if __name__ == "__main__":
    main()
