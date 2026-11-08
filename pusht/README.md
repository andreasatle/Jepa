# Push-T: SIGReg vs the VICReg-style penalty

**Status: closed (2026-11-08).** The question: does LeWorldModel's SIGReg regulariser behave better than the VICReg-style covariance penalty from `ball_in_cup` (with the sqrt on the diagonal) on a harder world-model task? Ball-in-cup was a tie at the ceiling (96 vs 96), so this repeats the comparison on Push-T, where the planner has room to fail.

## Conclusion

Both regularisers reach about the same planning success when their weight is right (39 and 35 of 50, within noise for one seed), so neither is better at its best. The difference is in how they behave around it: SIGReg at the paper's weight worked out of the box, holds the embedding scale at its target whatever the weight, and degrades more gently when the weight is off (worst 21 of 50 over a tenfold range, against VICReg's 5). Consistent with LeJEPA's claim of a well-behaved objective, not a claim of superiority on any one task. One seed per setting: differences of a few puzzles are noise (binomial standard deviation about 3 at 35 of 50).

## Results

Seed 0, 50 test puzzles (start and goal 25 expert steps apart, from held-out episodes), 50 steps allowed. Floor: random actions solve 1. Ceiling: replaying the expert's actions solves 50.

| regulariser | weight | solved | prediction MSE | embedding std |
|---|---|---|---|---|
| SIGReg | 0.03 | **39/50** | 0.010 | 0.98 |
| SIGReg | 0.09 (LeWM's Push-T value) | **35/50** | 0.017 | 0.98 |
| SIGReg | 0.3 | 21/50 | 0.032 | 0.99 |
| VICReg-style | 0.0016 | **35/50** | 0.006 | 0.52 |
| VICReg-style | 0.005 | 21/50 | 0.014 | 0.59 |
| VICReg-style | 0.0156 (ball-in-cup's kappa = 1, converted) | 5/50 | 0.028 | 0.61 |

Both ranges span a factor of 10. Both methods prefer the smaller weights here: more weight on the regulariser, less on prediction. Paired on the same puzzles, SIGReg 0.09 against VICReg 0.0156 solved 31 puzzles only SIGReg solved and 1 only VICReg solved.

**Why VICReg at kappa = 1 failed: not collapse.** Its embeddings use more dimensions than SIGReg's (effective rank 121 of 128 against 58). The penalty dominated the loss (0.52 of 0.54), and its off-diagonal term cannot reach zero at batch 128 and 128 dimensions: even an exact Gaussian batch scores about 128 from sampling noise alone, and the only way to lower it is to shrink every dimension. So the embeddings settle at a spread of 0.5-0.6 instead of 1, and prediction gets a smaller share of the gradient. SIGReg compares characteristic functions after smoothing (`Claude outputs/epps-pulley-test-tutorial.md`, section 6), so its finite-batch noise floor is small and it has no such pull on the scale.

## Setup

- **Data:** LeWorldModel's Push-T expert set (`quentinll/lewm-pusht`, 18,685 episodes, 2.3 M frames), resized from 224 to 64 px (`prepare_data.py`); 5% of episodes held out (seed 0), and all test puzzles come from them.
- **Model (`model.py`), the same for both arms:** a CNN encoder (64 px to 4x4x256), linear to 128-d, then LeWM's projector (Linear, BatchNorm, GELU, Linear); a causal transformer predictor (depth 4) over up to 3 frames, each token the frame's embedding plus its action block's embedding; pred_proj as in LeWM.
- **Training (`train.py`):** windows of 4 frames 5 steps apart, teacher forcing; loss = prediction MSE + weight x regulariser over all 4 frames' embeddings; AdamW lr 3e-4, warmup then cosine, 30,000 steps of batch 128 (about 2 passes, 67 min on the M-series Mac); afterwards precise BatchNorm (statistics recomputed with weights frozen; stale statistics wrecked planning in `ball_in_cup_sigreg` run 2).
- **Evaluation (`evaluate.py`):** LeWM's CEM planner (300 samples, 30 iterations, top 30, horizon 5 blocks of 5 steps, replan after 25 steps), cost = squared distance to the goal frame's embedding; success is the simulator's own test (pusher and T within 20 px of the goal, angle within 20 degrees). Every model gets the same 50 puzzles and the same planner noise per puzzle.
- **Simulator:** `swm/PushT-v1` from stable-worldmodel; its renders match the dataset frames and replaying the data's actions reproduces its states.

**Departures from LeWorldModel,** to fit a 24 GB Mac: 64 px frames and a CNN instead of 224 px and ViT-Tiny; the predictor adds the action embedding instead of AdaLN; lr 3e-4; held-out episodes for the test puzzles. At 64 px the 20 px success radius is about 2.5 pixels, so the task is harder for our encoder than for LeWM's (which reports 96% on its protocol).

**Weights were picked on the test puzzles** (VICReg's 0.0016, SIGReg's 0.03), which flatters both best scores a little; the conclusion rests on the behaviour across weights, not on the best numbers.

## Running it

```sh
uv run python prepare_data.py                      # once: data/frames64.npy, data/meta.npz (needs the h5 download)
./run_all.sh 0                                     # train and evaluate both regularisers at their default weights
uv run python train.py --reg sigreg --seed 0 --weight 0.03
uv run python evaluate.py runs/sigreg-w0.03-s0/model.pt     # or 'random' / 'expert'
uv run python play.py                              # push the T yourself: the pusher follows the mouse
```

In `play.py` success also needs the pusher in the red ring (where the expert's pusher ended); the title bar shows the three errors.
