"""
Goal-reaching on Push-T, LeWM's protocol (config/eval/pusht.yaml): 50 start states from held-out episodes,
the goal is the state 25 env steps later in the same episode, 50 env steps allowed. Success: the
simulator's own test (agent and block within 20 px, block angle within pi/9 of the goal), checked after
every step.

Planner (LeWM's CEM): action blocks of FRAMESKIP = 5 steps, horizon 5 blocks, 300 samples, 30 iterations,
the 30 best kept, starting from N(0, 1) in standardised action space; cost = squared distance between the
predicted embedding after the 5th block and the goal frame's embedding. All 5 blocks (25 steps) are
executed, then the planner runs again from the new frame (receding horizon 5), so 2 plans per episode.
The predictor sees up to HISTORY = 3 past frames.

    uv run python evaluate.py runs/sigreg-s0/model.pt      (or 'random' / 'expert' for the floor and the ceiling)

Every model gets the same 50 starts (seed 42) and the same CEM noise per start.
"""

import argparse
import sys

import cv2
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)
import torch

from model import FRAMESKIP, HISTORY, Encoder, Predictor
from train import Windows, to_input

N_EVAL, GOAL_OFFSET, BUDGET = 50, 25, 50
HORIZON, SAMPLES, ITERS, TOPK = 5, 300, 30, 30


def starts(meta) -> list[tuple[int, int]]:
    """(row of the start, row of the goal) for N_EVAL random starts in held-out episodes."""
    rows = np.concatenate([meta["ep_offset"][e] + np.arange(meta["ep_len"][e] - GOAL_OFFSET)
                           for e in meta["held_out_episodes"]])
    pick = np.sort(np.random.default_rng(42).choice(rows, N_EVAL, replace=False))
    return [(int(r), int(r) + GOAL_OFFSET) for r in pick]


def frame64(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))


class Planner:
    def __init__(self, ckpt: str, mu: np.ndarray, sd: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu")
        self.E, self.P = Encoder(), Predictor()
        self.E.load_state_dict(ck["E"])
        self.P.load_state_dict(ck["P"])
        self.E.to(self.device).eval()
        self.P.to(self.device).eval()
        self.mu, self.sd = mu, sd
        self.lo, self.hi = (lo - mu) / sd, (hi - mu) / sd           # action limits, standardised

    @torch.no_grad()
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.E(to_input(frame[None], self.device))[0]

    @torch.no_grad()
    def plan(self, z_hist: list, a_hist: list, z_goal: torch.Tensor, gen: torch.Generator) -> np.ndarray:
        """z_hist: past frame embeddings (last one = now); a_hist: the blocks taken after all but the last.
        Returns HORIZON raw action blocks, (HORIZON, FRAMESKIP, 2)."""
        dev = self.device
        lo, hi = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (self.lo, self.hi))
        mean = torch.zeros(HORIZON, 2 * FRAMESKIP, device=dev)
        std = torch.ones(HORIZON, 2 * FRAMESKIP, device=dev)
        z0 = torch.stack(z_hist)[None].expand(SAMPLES, -1, -1)
        a0 = (torch.stack(a_hist)[None].expand(SAMPLES, -1, -1) if a_hist
              else torch.zeros(SAMPLES, 0, 2 * FRAMESKIP, device=dev))
        for _ in range(ITERS):
            noise = torch.randn(SAMPLES, HORIZON, 2 * FRAMESKIP, generator=gen).to(dev)
            cand = torch.maximum(torch.minimum(mean + std * noise, hi), lo)
            z, a = z0, a0
            for h in range(HORIZON):
                a = torch.cat([a, cand[:, h:h + 1]], 1)
                zt, at = z[:, -HISTORY:], a[:, -HISTORY:]
                z = torch.cat([z, self.P(zt, at)[:, -1:]], 1)
            cost = (z[:, -1] - z_goal).pow(2).sum(-1)
            elite = cand[cost.topk(TOPK, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0)
        blocks = mean.cpu().numpy() * self.sd + self.mu
        return blocks.reshape(HORIZON, FRAMESKIP, 2)


def run_episode(env, policy, s_row, g_row, meta, frames, planner, trial) -> bool:
    state, goal = meta["state"][s_row].astype(np.float64), meta["state"][g_row].astype(np.float64)
    env.reset(seed=trial, options={"state": state, "goal_state": goal})
    if policy == "expert":
        for t in range(GOAL_OFFSET):
            _, _, done, _, _ = env.step(meta["action"][s_row + t].astype(np.float64))
            if done:
                return True
        return False
    rng = np.random.default_rng(trial)
    gen = torch.Generator().manual_seed(trial)
    z_goal = planner.encode(torch.from_numpy(np.array(frames[g_row]))) if planner else None
    z_hist, a_hist = [], []
    steps = 0
    while steps < BUDGET:
        if policy == "random":
            blocks = rng.uniform(meta["action"].min(0), meta["action"].max(0), (HORIZON, FRAMESKIP, 2))
        else:
            z_hist.append(planner.encode(frame64(env.render())))
            blocks = planner.plan(z_hist[-HISTORY:], a_hist[-(HISTORY - 1):] if len(z_hist) > 1 else [],
                                  z_goal, gen)
        for h, block in enumerate(blocks):
            for a in block:
                _, _, done, _, _ = env.step(a.astype(np.float64))
                steps += 1
                if done:
                    return True
                if steps >= BUDGET:
                    return False
            if planner and h < HORIZON - 1:     # the frame after each executed block joins the history
                std_block = torch.tensor(((block.reshape(-1) - planner.mu) / planner.sd), dtype=torch.float32,
                                         device=planner.device)
                a_hist.append(std_block)
                z_hist.append(planner.encode(frame64(env.render())))
        if planner:                             # the last block executed before replanning
            a_hist.append(torch.tensor(((blocks[-1].reshape(-1) - planner.mu) / planner.sd),
                                       dtype=torch.float32, device=planner.device))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="a model.pt, or 'random' or 'expert'")
    args = ap.parse_args()
    meta = dict(np.load("data/meta.npz"))
    frames = np.load("data/frames64.npy", mmap_mode="r")
    mu, sd = Windows.action_stats(meta)
    a = meta["action"]
    planner = None
    if args.policy not in ("random", "expert"):
        planner = Planner(args.policy, mu, sd, np.tile(a.min(0), FRAMESKIP), np.tile(a.max(0), FRAMESKIP))
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    wins = []
    for trial, (s, g) in enumerate(starts(meta)):
        ok = run_episode(env, args.policy, s, g, meta, frames, planner, trial)
        wins.append(ok)
        print(f"trial {trial:2d}: {'SOLVED' if ok else 'not solved'}", flush=True)
    print(f"success rate: {sum(wins)}/{N_EVAL}  ({args.policy})")


if __name__ == "__main__":
    sys.exit(main())
