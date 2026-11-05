# Jepa workspace

A container for self-contained experiment projects that share one Python
environment (`pyproject.toml`, `uv.lock`, `.venv` at the root).

Each project lives in its own top-level directory with **all** of its own
code, data, checkpoints, logs and generated media inside it — nothing
project-specific sits at the root. Run a project's scripts from inside its
directory; paths in the code are relative to the current directory.

## Projects

- [`ball_in_cup/`](ball_in_cup/README.md) — JEPA world-model pretraining,
  representation probing, and CEM/MPC planning on dm_control's
  ball-in-cup catch task.

- [`i_jepa_restart/`](i_jepa_restart/PLAN.md) — self-supervised learning on PathMNIST with LeJEPA (no EMA), one validated step at a time; started as a clean restart of I-JEPA, which stays there as the reference. `i_jepa/` is the earlier I-JEPA attempt, kept for reference.
