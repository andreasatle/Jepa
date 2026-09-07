# Use cases for Jepa

Ranked by how well JEPA's specific bet — predict in latent space, self-supervised, don't reconstruct the observation — matches real demand and existing evidence:

1. Robot manipulation and control from vision. Pretrain a video world model on large amounts of passive video, add action-conditioning from a small amount of robot interaction, then plan by rolling the latent forward. This is the intended flagship use and the reason the architecture exists: robot data is scarce, video is abundant. V-JEPA 2-AC already showed zero-shot pick-and-place. Highest-conviction, but lab-scale to do from scratch.

2. Self-supervised visual backbones where labels are scarce. Medical imaging, satellite/remote sensing, pathology, industrial inspection — huge unlabeled corpora, expensive labels. Use I-JEPA/V-JEPA to pretrain an encoder, then a light head with few labels. Proven competitive with MAE/DINO, and here you keep only E and throw P away — no agent involved. Most immediately practical, and the most solo-feasible.

3. Predictive monitoring / anomaly detection on high-dimensional telemetry. Train E+P on normal operation of a system — turbine vibration, datacenter metrics, network traffic, ICU vitals — and use latent prediction error as the anomaly score. JEPA's habit of ignoring unpredictable noise gives fewer false alarms than reconstruction-based detectors. Underhyped, large industrial market (predictive maintenance), tractable at small scale.

4. Video understanding at scale. Action recognition, temporal localization, retrieval, video-level anomaly/safety monitoring, all on top of a frozen V-JEPA encoder with small task heads. Deployable in content moderation, sports analytics, surveillance. Representation branch again — mature enough to build on now.

5. Autonomous-driving world models. Motion forecasting for other road users, planning, and closed-loop scenario/simulation generation from dashcam video, plus perception pretraining. Latent rollouts are cheaper than pixel-space ones (Wayve/comma-style), which matters at fleet scale. Big money, but crowded and safety-gated.

Honorable mention: neural simulation surrogates (fluids, weather, molecular dynamics) — learn latent dynamics of an expensive simulator so others run cheap rollouts. Less crowded than driving, and genuinely doable solo if you have a simulator to distill.