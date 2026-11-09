# Video JEPA on Push-T: plan

**Goal (the user, 2026-11-09):** understand V-JEPA, a JEPA on video, by building a small one end to end. Not a benchmark: the point is to see each mechanism work (or fail) at a size the Mac can train in hours.

**Status: planned, nothing built yet.**

## What the papers say

Three papers, read 2026-11-09 (V-JEPA's and V-JEPA 2's appendix tables only partly verbatim; values marked *unverified* are to be checked against the code before use):

- **V-JEPA** (Bardes et al. 2024, arXiv 2404.08471). Hide most of a clip and predict the *features* of the hidden part from the visible part. Clip of 16 frames, frame skip 4; patches of 16×16 px spanning 2 frames (tubelets); masks are spatial blocks repeated through all frames (so the model cannot copy from a neighbouring frame): short-range, 8 blocks covering 15% of each frame, and long-range, 2 blocks covering 70%, aspect 0.75–1.5, about 90% hidden on average. Context encoder sees the visible tokens; a narrow predictor (12 blocks, width 384) gets them plus mask tokens at the hidden positions and outputs one feature per hidden token; the target is the hidden tokens' features from an EMA copy of the encoder that saw the whole clip, with stop-gradient; loss L1 ("more stable" than L2, no numbers). EMA momentum 0.998 → 1.0 *unverified*. Smallest model ViT-L (~200 M); 90k iterations at batch 3072. Evaluation: frozen encoder, attentive probe (one learnable query cross-attending over all tokens, then an MLP and a linear layer). Ablations that matter here: features beat pixels (K400 73.7 vs 68.6); multi-block masks beat random tubes by a lot (K400 72.9 vs 51.5); attentive beats linear probing by about 17 points.
- **V-JEPA 2** (Assran et al. 2025, arXiv 2506.09985). The same objective scaled up (1 M hours of video, ViT-g, 3D RoPE, longer and larger clips at the end). New: **V-JEPA 2-AC**, an action-conditioned predictor trained on the *frozen* encoder with 62 hours of robot video; block-causal (a frame's tokens see the current and past frames, actions and arm states); planning by CEM on the L1 distance to the goal image's embedding. The world-model half of what we did on Push-T, with a video encoder underneath.
- **LeVJEPA** (Kuhn, Maes, …, LeCun, Balestriero, Buettner, Aug 2026, arXiv 2608.27395). LeJEPA's recipe on video: no EMA, no predictor, no stop-gradient. One global view (the whole clip at 224 px) and V = 4 local views (the same 16 frames, spatially cropped to 96 px, photometrically augmented); loss = mean squared distance of each local view's embedding to the global one + λ·SIGReg with λ = 0.02 (fixed, never tuned), on a projector output (Linear → 2048 → BatchNorm → GELU → 256), per view; SIGReg with 1,024 directions, 17 knots on [0, 3], weight e^{−t²/2} (LeWM's). The embedding is a [cls] token. Per-frame patches (tubelet 1), factorized 3D RoPE, block-causal attention (no measurable cost), and **95% of patch tokens dropped at random** before the encoder, which both cuts cost and raises accuracy (IN1K 33.9 → 47.6). Random dropping beats tube dropping (50.7 vs 39.6). Claims: matches or beats V-JEPA 2 at 5–20× less compute. A ViT-Tiny trained 12 h on one 16 GB consumer GPU at batch 128 (< 8 GB); V-JEPA with the same encoder fits only batch 28. Not stated: optimiser, learning rate, crop scales, augmentations. The paper reports its default configuration as 47.6 in one place and 50.7 in another.

**The two mechanisms differ in what they ask of the encoder.** V-JEPA: *fill in what is hidden* (predict hidden space-time from visible space-time; needs the EMA teacher to avoid collapse). LeVJEPA: *recognise the same clip from a crop* (invariance across views; SIGReg avoids collapse). Both are worth building: the first is V-JEPA itself, the second continues our SIGReg line (PathMNIST, ball-in-cup, Push-T).

## The task

**Data: the Push-T videos already on disk** (`pusht/data/frames64.npy`, 18,685 expert episodes, 49–246 frames each, 64 px; the same 5% held-out episodes). No download, and the simulator's state gives exact answers to probe for, which a video of people doesn't.

**The question the probes ask: does the video encoder know motion?** A single frame shows where things are, never how fast they move (ball-in-cup's lesson). The state has the pusher's position and velocity and the T's position and angle; the T's velocity and angular velocity come from differences of consecutive states. So, for the last frame of a held-out clip, probe for:

- **position** (pusher x, y; T x, y; T angle as sin, cos): an image encoder can do this; the floor.
- **velocity** (pusher; T; T's angular velocity): only something that looked at several frames can. This is what video pretraining has to earn.
- **contact**: is the pusher touching the T (from the state geometry)? A rough "is something happening" signal.

Probes are fitted on frozen features of training episodes and scored (R²) on held-out episodes: an attentive probe (as in the papers) and a linear probe on mean-pooled tokens.

**Baselines, to read the numbers:**

1. **Random-init encoder** with the same probes: the floor (PathMNIST's lesson: a random ViT already scores well on some things).
2. **The Push-T image encoder** (`pusht/runs/sigreg-s0`, SIGReg at 0.09) applied per frame, probing on the last two or three frames' embeddings together: the honest image-based answer for velocity, since frame differences can carry it.

## Design (Mac-sized)

- **Clips:** 16 frames, every 2nd env step (32 steps, about a quarter of a median episode), 64 px.
- **Encoder (shared by both arms):** a small ViT, width 192, depth 6–8, 3 heads; patches 8×8 px. V-JEPA arm: tubelets of 2 frames, 8 × 8 × 8 = 512 tokens per clip. LeVJEPA arm: tubelet 1, 1,024 tokens before dropping. 3D sin-cos or RoPE positions (start with sin-cos; RoPE only if needed).
- **Arm A, V-JEPA:** multi-block masks as in the paper, scaled to an 8×8 patch grid (short-range 8 blocks / 15%, long-range 2 blocks / 70%); predictor width 96, depth 4; L1 to the EMA target's layer outputs at the hidden tokens; EMA 0.998 → 1.0.
- **Arm B, LeVJEPA:** global view = the whole clip at 64 px; 4 local views = the same 16 frames cropped (scale to be chosen, about 0.3–0.6 of the area) and resized to 32 px; 95% random token drop; [cls] → projector (hidden 2048, out 256) → invariance + 0.02 · SIGReg. Augmentations: Push-T's colours are fixed and carry the meaning (green goal, grey T, blue pusher), so start with crops only and note it as a departure.
- **Training:** AdamW, warmup then cosine, batch as large as fits; target about 2–4 hours per run on the Mac (measure first). Same data order and seed per arm.
- **Out of scope for now:** V-JEPA 2-AC planning (step 6 is optional, after the encoders work), UCF101 or other real video, 3D RoPE, progressive resolution.

## Steps

Each step ends with a check that must pass before the next (`checks/`, as in `jepa_pathmnist`).

1. **Clips and labels.** A dataset of 16-frame clips from training episodes and a held-out one; probe targets (position, velocity, angular velocity, contact) from the state. Check: velocities from state differences match the recorded pusher velocity; a clip's frames are 2 steps apart; no held-out episode in training.
2. **Encoder and probes, untrained.** The ViT, the attentive and linear probes, the probe training loop. Check: shapes and token counts; the probes fit the targets from *ground-truth* state features almost perfectly (the harness works); the random-init floor for every target.
3. **Image baseline.** The Push-T image encoder per frame, probed on the last frames. Check: position close to `pusht`'s own quality; record velocity R².
4. **Arm A: V-JEPA.** Masks, predictor, EMA, L1. Check first: mask coverage matches the targets (about 90% hidden, blocks through all frames); an untrained model's loss is finite and the EMA moves. Short run (about 20 minutes) for collapse (feature std) and loss trend, then a full run; probe it.
5. **Arm B: LeVJEPA.** Views, token dropping, projector, SIGReg. Check: views are crops of the same clip; about 5% of tokens kept; SIGReg values on known samples (reuse `jepa_pathmnist/checks/check_lejepa.py`'s cases). Short run, then full; probe it.
6. **Compare and conclude.** One table: floor, image baseline, V-JEPA, LeVJEPA, on position, velocity, contact; plus the cost of each run. Optional afterwards: plan Push-T with a frozen video encoder and an action-conditioned predictor (V-JEPA 2-AC style), against the image encoder's 35/50.

## Risks and what would change the plan

- **Push-T is visually simple** (flat colours, three objects). An encoder might learn position quickly and velocity never, or the task might be too easy to tell the arms apart. If both arms max out every probe, the next data is a real-video set (UCF101 at 64 px), probed for action class.
- **V-JEPA without many tokens:** at 512 tokens with 90% hidden, the context is about 50 tokens; the target blocks might be guessable from one frame. If so, try 32-frame clips or smaller patches.
- **Speed:** 512–1,024 tokens per clip through a ViT on MPS; measure tokens per second in step 2 and size the runs from it.

## Decisions made during implementation

(none yet)
