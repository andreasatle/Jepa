"""
The Push-T world model: one architecture for both arms, so the regulariser is the only difference.

Shaped after LeWorldModel (arXiv 2603.19312, github.com/lucas-maes/le-wm), scaled down for a 24 GB Mac:

  encoder    CNN on a 64 px frame (ball_in_cup's: 64 -> 4x4x256), linear to D, then the projector:
             Linear -> BatchNorm -> GELU -> Linear (LeWM's MLP with BatchNorm, hidden 512 instead of 2048)
  predictor  a small causal transformer over up to HISTORY frames: token i is z_i plus the embedding of
             the action block taken after frame i, and output i predicts z_{i+1} (LeWM's ARPredictor
             conditions with AdaLN; adding the action embedding is the simpler form). Output goes through
             pred_proj, the same MLP as the projector, as in LeWM.
  regs       sigreg: LeWM's SIGReg exactly (17 knots on [0, 3], weight exp(-t^2/2), 1024 directions),
             per time step, averaged.
             vicreg: the user's ball_in_cup penalty (squared off-diagonal covariances plus
             (sqrt(var) - 1)^2 on the diagonal), per time step, averaged.

An action block is FRAMESKIP = 5 consecutive 2-d actions, flattened to 10 numbers, standardised with the
training set's mean and std.
"""

import torch
from torch import nn

D = 128               # embedding size
HISTORY = 3           # frames the predictor sees (LeWM: history_size 3)
FRAMESKIP = 5         # env steps per frame transition (LeWM: 5)
ACTION_DIM = 2 * FRAMESKIP


class MLP(nn.Sequential):
    """LeWM's projector / pred_proj: Linear -> BatchNorm -> GELU -> Linear."""

    def __init__(self, dim: int = D, hidden: int = 512):
        super().__init__(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, dim))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, D),
        )
        self.projector = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 64, 64) float in [0, 1] -> (N, D)."""
        return self.projector(self.net(x))


class Predictor(nn.Module):
    def __init__(self, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.action_embed = nn.Sequential(nn.Linear(ACTION_DIM, 4 * D), nn.SiLU(), nn.Linear(4 * D, D))
        self.pos = nn.Parameter(torch.zeros(1, HISTORY, D))
        layer = nn.TransformerEncoderLayer(D, heads, 4 * D, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(D)
        self.pred_proj = MLP()

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) frames, a: (B, T, ACTION_DIM) the block after each frame, T <= HISTORY.
        Returns (B, T, D): output i is the predicted embedding of frame i + 1."""
        b, t, _ = z.shape
        x = z + self.action_embed(a) + self.pos[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=z.device), diagonal=1)
        h = self.norm(self.blocks(x, mask=mask, is_causal=True))
        return self.pred_proj(h.reshape(b * t, D)).reshape(b, t, D)


class SIGReg(nn.Module):
    """LeWM's module.SIGReg: Epps-Pulley against N(0, 1) along random directions, times the batch size."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (T, B, D) -> scalar, averaged over directions and time."""
        a = torch.randn(z.size(-1), self.num_proj, device=z.device)
        a = a / a.norm(dim=0)
        xt = (z @ a).unsqueeze(-1) * self.t
        err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
        return ((err @ self.weights) * z.size(-2)).mean()


class VICRegPenalty(nn.Module):
    """The user's ball_in_cup penalty (jepa_train.cov_penalty_term), per time step, averaged over time."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (T, B, D) -> scalar."""
        terms = []
        for zt in z:
            c = zt - zt.mean(0, keepdim=True)
            cov = c.T @ c / (len(zt) - 1)
            diag = torch.diagonal(cov)
            off = cov - torch.diag(diag)
            terms.append(off.pow(2).sum() + ((diag + 1e-8).sqrt() - 1).pow(2).sum())
        return torch.stack(terms).mean()


# Weights. SIGReg: LeWM's Push-T value. VICReg: the user's kappa = 1 from ball_in_cup, carried over to a
# per-dimension mean MSE. ball_in_cup's loss was 0.5 * sum_d err^2 + kappa/2 * (two penalty terms), i.e.
# (D/2) * mse + kappa * (mean penalty); dividing by D/2 gives mse + (2 kappa / D) * penalty.
WEIGHTS = {"sigreg": 0.09, "vicreg": 2 * 1.0 / D}


def make_reg(name: str) -> nn.Module:
    return SIGReg() if name == "sigreg" else VICRegPenalty()
