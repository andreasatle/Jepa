# The algorithms in video_jepa, and what each one found

Written by Claude on 2026-11-14 from the code in `video_jepa/` and its `PLAN.md`, for reading. Everything here runs on Push-T videos at 64 px, on the Mac.

## 1. The data: clips and what we ask about them (`data.py`)

The source is the Push-T expert set already prepared for `pusht/`: 18,685 episodes, 49-246 frames each, 64 x 64 px, with the simulator's state for every frame. 5% of the episodes are held out, the same ones `pusht` holds out.

A **clip** is 16 frames, 2 env steps apart, so it spans 30 steps, 3 seconds at 10 Hz, all inside one episode. It is named by its last frame, and the labels describe that frame:

- **position:** the pusher's x, y; the T's x, y; the T's angle as sin and cos. One frame shows these.
- **velocity:** the pusher's vx, vy; the T's vx, vy and angular velocity, as differences over one clip step (0.2 s). Only something that looked at several frames can know these.
- **contact:** whether the pusher touches the T, from the geometry of the two shapes.

The question all of it asks: **does a video encoder learn motion, which a single image can't show?**

## 2. The encoder both methods share (`model.py`)

A small Vision Transformer over space and time: width 192, depth 6, 3 heads, 2.7 M parameters.

- **Patches:** each frame is cut into 8 x 8 px patches, an 8 x 8 grid. V-JEPA groups 2 frames into one token (a *tubelet*): 8 time slots x 64 = 512 tokens per clip. LeVJEPA keeps one frame per token: 16 x 64 = 1,024 tokens.
- **Positions:** fixed sine-cosine codes, a third of the channels each for time, row and column, added before any token is dropped, so a kept token still knows where and when it sits.
- **Dropping tokens:** the encoder can run on any subset of tokens (`keep_idx`), which is how V-JEPA's context encoder sees only the visible part and how LeVJEPA drops 95%. Padding, for batches of unequal length, is masked so no token attends to it.
- **[cls]:** LeVJEPA adds one learned summary token in front; its output is the clip's embedding.

## 3. V-JEPA: predict the hidden part (`masks.py`, `vjepa.py`)

The idea (Bardes et al. 2024): hide most of a clip, and predict the *features* of the hidden part from the visible part, not its pixels.

**Masks.** Spatial blocks, repeated through every time slot, so the model can't copy a hidden patch from the same place one frame earlier. Two kinds per step: short-range, 8 blocks of about 15% of the frame each, and long-range, 2 blocks of about 70% each. On our 8 x 8 grid they hide about 66% and 81% (the paper's 90% is for a finer grid). What is hidden is the *target*, the rest is the *context*.

**The loss**, for each mask kind:

- the **target encoder** sees the whole clip; its features at the hidden positions, layer-normalised, are the targets $h$ (no gradient);
- the **context encoder** sees only the visible tokens, giving $z$;
- a narrow **predictor** (width 96, depth 4) gets $z$ plus one mask token per hidden position and outputs a prediction $p$ for each;
- $\text{loss} = \text{mean}\,|p - h| / 2$, an L1 distance, as in the paper.

**Why it doesn't collapse.** If both encoders output a constant, the loss is zero. V-JEPA prevents this with an **EMA teacher**: the target encoder isn't trained, it follows the context encoder slowly, $\theta_{\text{target}} \leftarrow m\,\theta_{\text{target}} + (1-m)\,\theta_{\text{context}}$, with $m$ ramped from 0.998 to 1.0. Plus a stop-gradient on the targets.

**What we saw.** Most of the loss is position: a token's target can largely be guessed from where it is, and 40% of tokens are white background. The EMA target barely moved over a short run, and only started learning after about 1,400 steps of the full run (8,000 steps, 130 min).

## 4. LeVJEPA: recognise the same clip from a crop (`levjepa.py`)

The idea (Kuhn et al. 2026, LeJEPA's recipe on video): no masking, no predictor, no EMA teacher. Instead, *views* of the same clip must agree, and a regulariser keeps the embeddings spread out.

**Views.** One global view: the whole clip with 5% of its 1,024 tokens kept at random (51 tokens). Four local views: an s x s patch window of the frame (s = 5 or 6, 39-56% of the area), the same window in every frame, with 5% of its tokens kept (20 or 29). Each view goes through the encoder; its [cls] output goes through a projector (Linear to 2048, BatchNorm, GELU, Linear to 256), giving $z_v$.

**The loss:**

$$\text{loss} = (1-\lambda)\,\underbrace{\text{mean}_{\text{local } v}\,\|z_v - z_{\text{global}}\|^2}_{\text{invariance}} + \lambda\,\underbrace{\text{mean}_v\,\text{SIGReg}(z_v)}_{\text{spread}}, \qquad \lambda = 0.02$$

**SIGReg** keeps the batch of embeddings close to a standard Gaussian, which a collapsed batch is not. It projects the embeddings onto 1,024 random directions and, for each, compares the empirical characteristic function $\hat\varphi(t) = \text{mean}_j\, e^{i t x_j}$ with the Gaussian's $e^{-t^2/2}$ (the Epps-Pulley test, see `epps-pulley-test-tutorial.md`), weighted by $e^{-t^2/2}$ on $t \in [0, 3]$. A collapsed batch scores high, a Gaussian one near its small expected value. So collapse is a bad score, not a path the training happens to avoid.

**Departures from the paper:** crops cut on the patch grid at full resolution (the paper resizes them), no colour augmentation (Push-T's colours carry meaning), bidirectional attention.

**What we saw.** A short run (1,500 steps) learned position but no velocity at all; by step 5,000 of the full run (9,000 steps, 122 min) velocity had come. Slower to start than V-JEPA, twice as fast per clip.

## 5. Probes: what does a frozen encoder know? (`probes.py`)

The encoder is frozen; small models are fitted on top of it, on 8,000 training clips, and scored as R² on 2,000 held-out clips (1 = perfect, 0 = guessing the mean).

- **Linear probe:** ridge regression on the mean of all tokens. Averaging forgets where each token was, so it measures what is in the tokens' content.
- **Attentive probe** (as in V-JEPA): one learned query attends over all tokens, then an MLP; it can look at different frames through different heads. This is the main measure.
- **The floor:** the same probes on a random, untrained encoder. It already knows a fair amount of position (R² about 0.56-0.60), so every number is read against it.
- **Noise:** changing only the probe's random seed moves velocity by up to 0.08 and contact by up to 0.15. Smaller differences aren't results.

## 6. The image baseline (`baseline_image.py`)

The encoder of the `pusht` world model (a CNN trained with SIGReg to predict the next frame's embedding from the action), applied to each frame separately; the probes then read the 16 frames' embeddings side by side. A strong baseline: trained on 1.6 M frames, with actions.

## 7. The comparison (step 6)

Attentive probe, three probe seeds:

| encoder | position | velocity | contact | contact accuracy |
|---|---|---|---|---|
| random (floor) | 0.56-0.60 | 0.22-0.25 | 0.23-0.27 | 71-73% |
| V-JEPA | 0.71-0.75 | 0.43-0.54 | 0.50-0.59 | 83-86% |
| LeVJEPA | 0.84-0.87 | 0.40-0.48 | 0.38-0.53 | 78-84% |
| image baseline | 0.84 | 0.75-0.77 | 0.68-0.69 | 91-92% |

Both video methods learn motion from video alone, without labels or actions. Neither reaches the image baseline on motion; their weak point is the pusher's velocity (a 4 px dot inside an 8 px patch). LeVJEPA is clearly better on position; on velocity and contact the two are within the probe's noise.

## 8. Planning with the frozen video encoder (`plan_ac.py`, step 7)

The idea (V-JEPA 2-AC, at our scale): keep the video encoder frozen, and learn only how actions change its embedding; then plan by searching for actions whose predicted outcome lands near the goal's embedding.

- **A moment** is LeVJEPA's [cls] embedding, standardised per dimension. Two variants: **still** (the frame repeated 16 times, as V-JEPA 2-AC encodes frames one at a time) and **clip** (the 3 seconds of video ending there, so the moment carries motion).
- **The predictor** is `pusht`'s: a small causal transformer over the last 3 moments, each plus the embedding of the 5-step action block taken after it; output $i$ predicts moment $i+1$. Trained with MSE on 450,850 frozen moments; no regulariser, since a frozen encoder can't collapse.
- **The planner** is `pusht`'s CEM (cross-entropy method): sample 300 sequences of 5 action blocks, roll each through the predictor, keep the 30 whose last predicted moment is closest to the goal's, refit the sampling distribution to them, repeat 30 times; execute the plan and replan once. Same 50 puzzles as `pusht` (start and goal 25 steps apart in held-out episodes, 50 steps allowed).

| planner | solved of 50 |
|---|---|
| random actions | 1 |
| LeVJEPA, still | 4 |
| LeVJEPA, clip | 5 |
| image world model (`pusht`), rerun as a control | 35 |

**Why it fails:** distances between LeVJEPA embeddings follow the true distances between states only roughly (rank correlation 0.5-0.6), and weakly among states that are already close (0.34-0.37). The image world model's are the opposite (0.18 overall, 0.49 among close pairs). Reaching a goal 2.5 pixels wide needs the fine distinctions, which a model trained to predict the effect of each push learns and a model trained to match crops doesn't.

**Not yet a fair verdict:** V-JEPA 2-AC plans over the encoder's patch tokens, not one summary vector. The probes showed the tokens hold much more position detail than the [cls] (0.84 against 0.61). The next test plans over tokens.

## The lessons so far

1. **Video alone teaches motion**, but a model trained to predict the effect of actions learns the details that matter for control better.
2. **What a representation is trained for decides what it keeps.** Crop invariance keeps the coarse layout; action-conditioned prediction keeps the small differences a push makes.
3. **A frozen probe and a planner measure different things.** LeVJEPA matched the image baseline on probed position, yet plans far worse, because planning needs distances that are precise near the goal, not only information a probe can extract.
