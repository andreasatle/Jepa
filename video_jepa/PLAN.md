# Video JEPA on Push-T: plan

**Goal (the user, 2026-11-09):** understand V-JEPA, a JEPA on video, by building a small one end to end. Not a benchmark: the point is to see each mechanism work (or fail) at a size the Mac can train in hours.

**Status: steps 1-5 done (2026-11-12); step 6 (compare, delete one arm) next.** Order (the user, 2026-11-11): V-JEPA first, then LeVJEPA; the user expects to use LeVJEPA in the end, and the comparison shows what it gains or loses.

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

1. [x] **Clips and labels** (`data.py`, `checks/check_data.py`). A dataset of 16-frame clips from training episodes and a held-out one; probe targets (position, velocity, angular velocity, contact) from the state. Check: velocities from state differences match the recorded pusher velocity; a clip's frames are 2 steps apart; no held-out episode in training.
2. [x] **Encoder and probes, untrained** (`model.py`, `probes.py`, `checks/check_model.py`, `checks/check_probes.py`). The ViT, the attentive and linear probes, the probe training loop. Check: shapes and token counts; the probes fit the targets from *ground-truth* state features almost perfectly (the harness works); the random-init floor for every target.
3. [x] **Image baseline** (`baseline_image.py`). The Push-T image encoder per frame, probed on the last frames. Check: position close to `pusht`'s own quality; record velocity R².
4. [x] **Arm A: V-JEPA** (`masks.py`, `vjepa.py`, `checks/check_vjepa.py`; run `runs/vjepa-s0`). Masks, predictor, EMA, L1. Check first: mask coverage matches the targets (about 90% hidden, blocks through all frames); an untrained model's loss is finite and the EMA moves. Short run (about 20 minutes) for collapse (feature std) and loss trend, then a full run; probe it.
5. [x] **Arm B: LeVJEPA.** Views, token dropping, projector, SIGReg. Check: views are crops of the same clip; about 5% of tokens kept; SIGReg values on known samples (reuse `jepa_pathmnist/checks/check_lejepa.py`'s cases). Short run, then full; probe it.
6. **Compare and conclude.** One table: floor, image baseline, V-JEPA, LeVJEPA, on position, velocity, contact; plus the cost of each run. **Then one arm is deleted** (rule agreed with the user, 2026-11-12, for lean code): LeVJEPA is kept if it is at least as good as V-JEPA on velocity and contact (attentive probe, within the spread of the two probe seeds), since it is the simpler recipe (no EMA target, no predictor, no masks) and the one the user expects to use; V-JEPA is kept only if it is better beyond that spread. The loser's files go in the commit that records the conclusion (V-JEPA: `vjepa.py`, `masks.py`, `checks/check_vjepa.py`, and the `valid` padding in `model.py` that only its masks need; LeVJEPA: its own files), its numbers stay in this plan and its code in git history. `baseline_image.py` stays only while a later step compares against it. Optional afterwards: plan Push-T with a frozen video encoder and an action-conditioned predictor (V-JEPA 2-AC style), against the image encoder's 35/50.

## Risks and what would change the plan

- **Push-T is visually simple** (flat colours, three objects). An encoder might learn position quickly and velocity never, or the task might be too easy to tell the arms apart. If both arms max out every probe, the next data is a real-video set (UCF101 at 64 px), probed for action class.
- **V-JEPA without many tokens:** at 512 tokens with 90% hidden, the context is about 50 tokens; the target blocks might be guessable from one frame. If so, try 32-frame clips or smaller patches.
- **Speed:** 512–1,024 tokens per clip through a ViT on MPS; measure tokens per second in step 2 and size the runs from it.

## Decisions made during implementation

- **Order (the user, 2026-11-11):** V-JEPA first, LeVJEPA second. LeJEPA beat I-JEPA on PathMNIST, so the user expects LeVJEPA to be the one used in the end; the open question is velocity, which a view-invariance objective with crops and 95% token dropping may discard and V-JEPA's prediction of hidden space-time may keep.
- **Velocity labels are differences over one clip step** (2 env steps, 0.2 s), in units per second, for the pusher and the T alike: what the clip's frames can show. The simulator's recorded pusher velocity is only a check: one-step differences track it at correlation 0.99 (recorded = 0.93 × difference; it is the end-of-step velocity under the PD controller), the two-step labels at 0.93.
- **Contact is geometric:** the pusher's centre within radius + 2 units of the T's two boxes (the simulator's add_tee, scale 30; pusher radius 15). 49% of frames; the T moves in 95% of contact frames and 4% of the others, so the label means what it says.
- **Frames and state are aligned at lag 0:** the blue pusher's centroid in the 64 px frame matches the state at 0.19 px rms (lag −1: 1.35, +1: 2.85), scale 0.1248 px per unit (64/512 = 0.125).
- **Clips:** 1,687,729 possible training clips from 17,751 episodes, 88,457 held-out from 934 (pusht's split). Probe sets are fixed seeded subsets (`Clips(..., n=, seed=)`); their sizes are set in step 2.
- **Encoder:** width 192, depth 6, 3 heads, 8 px patches, 2.7 M parameters; V-JEPA's tubelets of 2 give 512 tokens, LeVJEPA's single frames 1,024. 3-D sin-cos positions (a third of the channels each for time, row, column). Pixels normalised with Push-T's own statistics (mean about 0.97: a white background), not ImageNet's.
- **Speed on the Mac (MPS), forward + backward, batch 32:** 172 clips/s with all 512 tokens, 750 clips/s with 51 kept. A V-JEPA step (target forward on all tokens, context forward and backward on about 10%, predictor) should run at roughly 150-200 clips/s, so 2 hours is about one pass over the 1.69 M training clips.
- **Probes** (`probes.py`): 8,000 training and 2,000 held-out clips (seeds 1, 2), fixed for every arm. Linear: ridge (weight 1) on the mean of the tokens. Attentive: one query, 4 heads, residual MLP, linear head; AdamW lr 1e-3, weight decay 0.05, 30 epochs, one-cycle schedule, MSE on standardised targets; one recipe, never tuned per encoder. Scored as R² on held-out clips, contact also as accuracy.
- **The harness works** (`checks/check_probes.py`): on tokens holding each frame's true state, the attentive probe gets position 0.999, velocity 0.964 (it subtracts the last two frames through different heads) and contact 94% accurate; a linear probe on the last frame's state gets position 1.000 and velocity 0.09, as it should with one frame.
- **The floor, random-init encoder** (two seeds; R², position / velocity / contact, contact accuracy):

  | probe | seed 0 | seed 1 |
  |---|---|---|
  | linear (mean) | 0.55 / 0.14 / 0.22, 71% | 0.55 / 0.14 / 0.19, 70% |
  | attentive | 0.59 / 0.22 / 0.25, 72% | 0.55 / 0.21 / 0.14, 66% |

  A random encoder already gives the T's position (R² about 0.8) and the pusher's (about 0.7), but not the T's angle (0.1-0.3) or the pusher's velocity (0.1). The attentive probe's spread between seeds is up to 0.1 on contact, so differences smaller than that are not results.
- **The image baseline** (`baseline_image.py`, the world model's encoder from `pusht/runs/sigreg-s0` on each frame; R², position / velocity / contact, contact accuracy):

  | probe | position | velocity | contact | accuracy |
  |---|---|---|---|---|
  | linear, last frame | 0.73 | 0.21 | 0.61 | 89% |
  | linear, last 3 frames | 0.74 | 0.67 | 0.63 | 90% |
  | attentive, 16 frames | 0.84 | 0.75 | 0.69 | 92% |

  x, y positions alone 0.94-0.95 (the check: > 0.85; my first check used the whole position group and failed on the T's angle, which this encoder holds only nonlinearly: sin, cos 0.3-0.4 linearly, 0.6-0.7 attentive). Velocity comes from comparing frames: 0.21 from one frame, 0.67 from three side by side; the T's angular velocity stays hard (0.13 linear, 0.67 attentive). This is a strong baseline: a trained encoder (1.6 M frames, action-conditioned prediction), so **the bar for the video encoders is velocity 0.67-0.75 and contact 90-92%**, against the random floor's 0.14-0.22 and 66-72%.
- **V-JEPA masks on an 8 x 8 grid** (`masks.py`): short-range hides 66% on average, long-range 81% (the paper's ~90% is for its 14 x 14 grid; our blocks round to whole patches). The block size is drawn once per batch, as in the official collators. Cutting every clip's lists to the batch's shortest (the official way) left 16 context tokens in long-range batches, because two 70% blocks sometimes cover nearly the whole 8 x 8 frame; so lists are padded instead, with attention and the loss ignoring the padding (checked: a clip's outputs are identical alone and padded).
- **Speed:** a V-JEPA step runs at 69 clips/s (compute-bound; data loading alone 3,700 clips/s), so 2 hours is about 8,000 steps at batch 64.
- **Short run** (`runs/short`, 1,500 steps, 25 min): loss 0.86 -> 0.09, no collapse (rank of pooled features 36 -> 23). Two findings before the full run:
  1. **Most of the loss is position.** Guessing each target token from its position alone gives L1 0.099; position explains 80% of the target features' variance at init (55% after the run), and 40% of tokens are pure white background. So the loss falling to 0.09 says little; content learning shows only in the probes.
  2. **The EMA target barely moved:** with momentum ramped 0.998 -> 1.0 over only 1,500 steps it stayed near its random init (content variance 15.5 -> 13.4), and its probes are at the floor (attentive position 0.53, velocity 0.10). The online encoder learned: content variance 15.5 -> 55.7, and its attentive probe gives position 0.66, velocity 0.40, contact 83% (floor 0.59 / 0.22 / 72%; image baseline 0.84 / 0.75 / 92%). So both encoders are probed from now on; V-JEPA's own evaluation reads the target encoder (to be confirmed in its code).
- **Full run** (`runs/vjepa-s0`, 8,000 steps, 130 min, 512 k clips, about a third of one pass): the target encoder starts to move around step 1,400; from then the target features' spread across clips grows about 20-fold (0.02 -> 0.55) and the loss rises with it (0.08 -> 0.36, because the targets now carry content), both flat from step 4,000. The rank of the pooled features settles at 7-9: low, but Push-T's state has about that many numbers, and the spread never shrinks, so no collapse.
- **V-JEPA's probes** (attentive; R², position / velocity / contact, accuracy). The linear probe on mean-pooled tokens stays at the floor for every encoder (0.55 / 0.15 / 0.13): averaging over all tokens, 40% of them background, keeps only the position code.

  | encoder | position | velocity | contact | accuracy |
  |---|---|---|---|---|
  | random floor | 0.55-0.59 | 0.21-0.22 | 0.14-0.25 | 66-72% |
  | V-JEPA, target (EMA) | 0.72 | 0.47 | 0.53 | 85% |
  | V-JEPA, online | 0.74 | 0.50 | 0.55 | 85% |
  | image baseline | 0.84 | 0.75 | 0.69 | 92% |

  Per target, V-JEPA is closest to the image baseline on the T (velocity 0.61-0.67 vs 0.75-0.77, angular velocity 0.40-0.47 vs 0.67) and furthest on the pusher's velocity (0.31-0.37 vs 0.78). The pusher is a 4 px dot in a 64 px frame, half of an 8 px patch, and moves about 2 px between the two frames of a tubelet, so a tubelet token sees little of its motion; the image encoder's CNN has full resolution. V-JEPA learned velocity from video alone (0.21 -> 0.50) but stays below a per-frame image encoder trained with actions on a world-model task, at a third of an epoch. The online and target encoders are within the probe's noise of each other.
- **LeVJEPA as built** (`levjepa.py`, `checks/check_levjepa.py`): one global view (5% of the 1,024 single-frame tokens) and 4 local views (an s x s patch window, s = 5 or 6 drawn per batch, the same window in every frame, 5% of its tokens); [cls] -> projector (2048, BatchNorm, GELU, 256); loss 0.98 * invariance + 0.02 * SIGReg (LeWM's). Departures: local views are patch-aligned windows at native resolution with their true sin-cos positions (the paper resizes 96 px crops and has RoPE); crops only, no colour augmentation; bidirectional attention. Optimiser from our PathMNIST LeJEPA (lr 5e-4, wd 0.05), batch 128. The shared encoder got an optional [cls] (`model.py`, off for V-JEPA). Checks: views stay inside one window and keep 5%; SIGReg on a collapsed sample equals its exact value (205.8 at batch 512) and a Gaussian sample is within 3% of its expectation (1.03 vs 1.05); the loss falls on one batch.
- **Speed:** 159 clips/s at batch 128 (five encoder passes of 20-51 tokens per clip; cutting the views takes 21 ms a batch), so 2 hours is about 9,000 steps, 1.15 M clips: twice V-JEPA's clips in the same time.
- **Short run** (`runs/le-short`, 1,500 steps, 20 min): no collapse (cls std 0.11 -> 0.74, rank 38 -> 23), loss 0.95 -> 0.28. But the probes learned almost nothing beyond position: attentive 0.67 / 0.22 / 0.20, 69% (floor 0.59 / 0.22 / 0.25, 72%; V-JEPA's short run, online encoder, 0.66 / 0.40, 83%). Velocity is exactly at the floor. Not a probe artefact: a linear probe on the [cls] alone gives 0.61 / 0.17 / 0.08, and on the [cls] with 5% of tokens kept, as in training, 0.50 / 0.08. The likely reason, to be tested by the full run: a crop and the whole clip agree most cheaply on where the T and the goal are, which barely changes over 3 s, so invariance needs no motion; and 5% of 1,024 tokens leaves about 3 per frame, too few to see a 4 px pusher move. The full run is at equal time with V-JEPA's, as planned.
- **The short run misled:** at step 5,000 of the full run the same recipe gives position 0.84, velocity 0.44, contact 81%. LeVJEPA needs more steps before it learns motion than V-JEPA does; the short run's explanation (invariance needs no motion) was wrong, or true only early.
- **Full run** (`runs/levjepa-s0`, 9,000 steps, 122 min, 1.15 M clips, about 0.7 of a pass): no collapse (cls std 0.75, rank 37 at the end), loss 0.95 -> 0.14-0.19.
- **Probe seeds:** `probes.py --probe-seed` now sets the attentive probe's init and data order (it was always 0, so step 6's "spread of the two probe seeds" had never been measured). Three probe seeds (0, 1, 2) per encoder, attentive; R², position / velocity / contact, accuracy:

  | encoder | position | velocity | contact | accuracy |
  |---|---|---|---|---|
  | V-JEPA, target | 0.71-0.72 | 0.43-0.51 | 0.50-0.59 | 83-86% |
  | V-JEPA, online | 0.73-0.75 | 0.46-0.54 | 0.54-0.56 | 84-86% |
  | LeVJEPA | 0.84-0.87 | 0.40-0.48 | 0.38-0.53 | 78-84% |

  The probe alone moves velocity by up to 0.08 and contact by up to 0.15 between seeds, more than the step-2 floor suggested. LeVJEPA is better on position beyond the spread (+0.1, the T's angle 0.72-0.78 vs 0.55-0.65); on velocity and contact its ranges overlap V-JEPA's, but its means are lower (velocity 0.43 vs 0.47-0.50; contact 0.43 vs 0.54-0.55; accuracy 80% vs 85%). Per target, LeVJEPA trails on the pusher's velocity (0.20-0.29 vs 0.30-0.41) and matches on the T's angular velocity (0.45-0.55 vs 0.35-0.47).

