# Seeded paper-style renderings — 2026-10-29

This package contains **30 recorded episodes and 120 output files**: five episodes for each of the 2v1, 5v2 and 7v3 scenarios in both training stages, with a PNG, GIF, CSV and PDF for every episode. The episodes use the checkpoints already supplied in this repository under `results/MPE/`; **no model was retrained** for this rendering run.

The static figures use the original GitHub renderer's drawing functions and the visual conventions of paper Figures 7–9. Success and breach labels are recorded separately in the results table and gallery, so the original figure legends remain unchanged. Failed episodes are retained.

Everything in this release is isolated under `reproductions/2026-10-29-renderings/`. Existing repository files and model checkpoints are preserved.

## View the published results

- [Episode gallery](render_output/index.html): thumbnails, verified outcomes and links to all four formats. Open this file locally after cloning or downloading the package; GitHub displays the HTML source rather than running the gallery.
- [Per-episode results](render_output/success_summary.csv): 30 rows, including seeds, capture counts, breach indicators and output paths.
- [Rendering manifest](render_output/render_manifest.json): provenance, completeness checks and output SHA256 hashes.
- [Computer configuration](hardware.json): the public hardware and execution configuration used for this run.

The published snapshot is in `render_output/`. Reproduction commands write to a separate `replay_output/` directory by default, preserving the snapshot.

All six first episodes below are verified successes. These direct image and PDF links can be viewed from GitHub:

| Paper panel | Scenario and stage | PNG | GIF | PDF |
|---|---|---|---|---|
| Figure 7(a) | 2v1, Stage I | [PNG](render_output/stage1/rmappo_2v1_stage1_ep01.png) | [GIF](render_output/stage1/rmappo_2v1_stage1_ep01.gif) | [PDF](render_output/stage1/rmappo_2v1_stage1_ep01.pdf) |
| Figure 7(b) | 2v1, Stage II | [PNG](render_output/stage2/rmappo_2v1_stage2_ep01.png) | [GIF](render_output/stage2/rmappo_2v1_stage2_ep01.gif) | [PDF](render_output/stage2/rmappo_2v1_stage2_ep01.pdf) |
| Figure 8(a) | 5v2, Stage I | [PNG](render_output/stage1/rmappo_5v2_stage1_ep01.png) | [GIF](render_output/stage1/rmappo_5v2_stage1_ep01.gif) | [PDF](render_output/stage1/rmappo_5v2_stage1_ep01.pdf) |
| Figure 8(b) | 5v2, Stage II | [PNG](render_output/stage2/rmappo_5v2_stage2_ep01.png) | [GIF](render_output/stage2/rmappo_5v2_stage2_ep01.gif) | [PDF](render_output/stage2/rmappo_5v2_stage2_ep01.pdf) |
| Figure 9(a) | 7v3, Stage I | [PNG](render_output/stage1/rmappo_7v3_stage1_ep01.png) | [GIF](render_output/stage1/rmappo_7v3_stage1_ep01.gif) | [PDF](render_output/stage1/rmappo_7v3_stage1_ep01.pdf) |
| Figure 9(b) | 7v3, Stage II | [PNG](render_output/stage2/rmappo_7v3_stage2_ep01.png) | [GIF](render_output/stage2/rmappo_7v3_stage2_ep01.gif) | [PDF](render_output/stage2/rmappo_7v3_stage2_ep01.pdf) |

## Reproduce the complete run

Run these commands from the **repository root** in a fresh checkout. Python 3.10 is required for the historical dependency versions:

```bash
python3.10 -m venv .venv-repro
.venv-repro/bin/python -m pip install -r reproductions/2026-10-29-renderings/requirements.txt

PYTHON=.venv-repro/bin/python \
  bash reproductions/2026-10-29-renderings/scripts/render_paper_suite.sh
```

If an environment with the required dependencies is already active:

```bash
PYTHON=python bash reproductions/2026-10-29-renderings/scripts/render_paper_suite.sh
```

The suite runs all six configurations, exports the four formats, and then validates and summarizes the outputs. Defaults are five episodes per configuration and the nominal paper sensing parameters. The six base seeds are listed below; episode `epXX` uses `base_seed + XX - 1`.

To use another output directory inside the package:

```bash
PYTHON=.venv-repro/bin/python \
  RENDER_ROOT=reproductions/2026-10-29-renderings/another_replay \
  bash reproductions/2026-10-29-renderings/scripts/render_paper_suite.sh
```

`PYTHON=...` selects the Python interpreter. `RENDER_ROOT` selects the output root for the complete suite. The package's ordinary [requirements.txt](requirements.txt) contains the direct historical version pins. To match the full recorded installed-version snapshot instead, use the pinned [requirements-lock.txt](requirements-lock.txt) in the installation step:

```bash
.venv-repro/bin/python -m pip install \
  -r reproductions/2026-10-29-renderings/requirements-lock.txt
```

## Run one configuration

For example, render the five Stage II 7v3 episodes:

```bash
PYTHON=.venv-repro/bin/python \
  ALGO=rmappo DEFENDERS=7 INTRUDERS=3 SEED=61003 EPISODES=5 \
  LEGEND_LOC=paper GIF=true CSV=true PDF=true \
  MODEL_DIR=results/MPE/protected_zone_stage2/rmappo/rmappo_7v3_stage2_/models \
  bash reproductions/2026-10-29-renderings/scripts/render_stage2.sh
```

The Stage I shell is `scripts/render_stage1.sh` in this package. Its three base seeds are all `61000`. Stage II base seeds are `61014` for 2v1, `61000` for 5v2, and `61003` for 7v3. Individual shells also accept `OUTPUT_DIR` for a custom stage output directory.

When a case is rendered again into the same output directory, its previous matching episode files and case summary are moved to that directory's `_archive/<timestamp_case>/` subdirectory. Other cases are retained.

To regenerate a replay's gallery and tables, or validate the published snapshot without writing it:

```bash
# Validate and summarize the default replay_output directory.
.venv-repro/bin/python \
  reproductions/2026-10-29-renderings/scripts/summarize_render_output.py

# Read-only validation of the published snapshot.
.venv-repro/bin/python \
  reproductions/2026-10-29-renderings/scripts/summarize_render_output.py \
  --render-root reproductions/2026-10-29-renderings/render_output \
  --validate-only
```

## Recorded outcomes

The base seeds were chosen to retain a verified successful first episode for each configuration. The next four consecutive seeds are also published, including every failure. These small rendering panels are **not the paper's training-window defense success rates**, a held-out benchmark, or evidence of an algorithm ranking.

| Stage | Scenario | Episode seeds | ep01 | ep02 | ep03 | ep04 | ep05 | Successful episodes |
|---|---|---|---|---|---|---|---|---|
| I | 2v1 | 61000–61004 | Success | Success | Success | Success | Success | 01, 02, 03, 04, 05 |
| I | 5v2 | 61000–61004 | Success | Success | Success | Success | Success | 01, 02, 03, 04, 05 |
| I | 7v3 | 61000–61004 | Success | Breach | Success | Breach | Success | 01, 03, 05 |
| II | 2v1 | 61014–61018 | Success | Breach | Breach | Breach | Breach | 01 |
| II | 5v2 | 61000–61004 | Success | Success | Success | Success | Success | 01, 02, 03, 04, 05 |
| II | 7v3 | 61003–61007 | Success | Breach | Breach | Success | Breach | 01, 04 |

There are **21 successful episodes and 9 breach episodes**, with no timeouts in this recorded panel. For example, Stage II 7v3 `ep03` eventually captures all three intruders but contains an earlier zone breach, so it is classified as a breach.

A verified success requires the original environment to return defense success, all intruders to be captured, capture-event distances to satisfy the capture radius, and no zone breach during the episode. The audit checks sampled positions, line segments between adjacent samples, sticky breach state and the terminal world state. The summary process independently cross-checks the CSV evidence and capture events.

Use `verified_defense_success=True` or `status=success` to identify successful episodes. A final capture count alone is insufficient.

## Files and model provenance

Each stage directory contains the following pattern for each scenario:

```text
render_output/stage1/rmappo_2v1_stage1_ep01.png
render_output/stage1/rmappo_2v1_stage1_ep01.gif
render_output/stage1/rmappo_2v1_stage1_ep01.csv
render_output/stage1/rmappo_2v1_stage1_ep01.pdf
render_output/stage1/rmappo_2v1_stage1_summary.json
```

The same pattern applies to `ep02`–`ep05`, the 5v2 and 7v3 scenarios, and `stage2`. Replay output uses the identical names beneath `replay_output/` or the selected `RENDER_ROOT`.

The model directory pattern, relative to the repository root, is:

```text
results/MPE/protected_zone_stageX/rmappo/rmappo_MvN_stageX_/models/
```

Stage I loads `actor.pt`, `critic.pt`, `intruder_actor.pt` and `intruder_critic.pt`. Stage II loads `defender_actor.pt`, `defender_critic.pt`, `intruder_actor.pt` and `intruder_critic.pt`. Model hashes and the actual runtime configuration are recorded in each `*_summary.json`.

The original execution source hashes are retained in each summary's `metadata.source`. The packaged scripts are namespaced copies of those execution scripts; packaging adapts imports, path discovery and English help text only. The existing repository core simulation files and model weights remain byte-for-byte unchanged.

**Native endpoint convention:** the original renderer records plot positions before each environment step. Its final square/triangle therefore marks the last recorded plotting frame, which can precede the audited terminal world state. The CSV terminal records and capture-event distances are the outcome evidence. A terminal nearest-defender distance is not the distance at the earlier capture event.

## Numerical settings and original figure style

| Setting | Value |
|---|---|
| World / protected-zone / capture radius | 5.0 / 0.5 / 0.2 m |
| Time step / episode horizon | 0.1 s / 300 steps |
| Defender / intruder maximum speed | 1.0 / 1.0 m/s |
| Stage II nominal AoA standard deviation / reference range | 0.0262 rad, approximately 1.5° / 0.5 m |
| CPF particles / acceleration process-noise standard deviation | 512 / 0.2 |
| Threat decay coefficient / prediction limit / scan step / switching time | 0.15 / 30 s / 1 s / 3 s |
| Confidence ellipse | 3-sigma, initial and final posterior |
| Static figure / PNG resolution | 6 × 6 inches / 300 dpi |
| Trajectory linewidth / marker-size argument | 2 / 6 |
| Legend / axis-label font size | 10 / 10 |

These numerical settings map to the paper's geometry, AoA, CPF and threat settings. The released environment dynamics, reward logic and native collection behavior are retained; this is not a claim that every paper equation has been separately reimplemented or validated. In particular, the original Stage II renderer retains an intruder sensing radius of **1.0 m** and at most **2 nearest neighbors**. Stage I ignores the defender's discrete assignment; Stage II uses the released greedy rebalancing. The native collectors do not apply the training runner's assignment masks.

The figures retain green dashed defender trajectories, blue solid intruder trajectories, the red protected-zone outline, the dashed world boundary, and the gray initial/final confidence ellipses. Defender start/end markers are circles/squares; intruder start/end markers are circles/triangles. Stage I has the original five legend entries at the upper right. Stage II adds `Confidence ellipse (3σ) start/end`, at the lower left for 2v1 and 7v3 and upper left for 5v2. Large ellipses are clipped at the original plotting axes, as in the paper.

## Computer and software configuration

| Component | Recorded configuration |
|---|---|
| Computer | Mac Studio, model identifier Mac15,14 |
| Processor | Apple M3 Ultra |
| CPU | 32 cores: 24 performance and 8 efficiency |
| GPU | 80 cores; not used for these runs |
| Unified memory | 512 GB |
| Architecture | arm64 |
| Operating system | macOS 27.0.1, build 26A434 |
| Execution device | CPU only |
| PyTorch threads | 1 intra-op thread and 1 inter-op thread |
| Rollout environments | 1 |
| Python | 3.10.15 |
| PyTorch / NumPy | 2.1.0 / 1.25.2 |
| Gymnasium / Matplotlib | 0.28.1 / 3.7.2 |

Hardware and software details are also available in [hardware.json](hardware.json) and the per-case summaries. This package includes general machine specifications only, without device serial numbers or unique device identifiers.

Fixed seeds, identical model hashes and matching dependencies support repeatability. Even CPU-only runs can differ across architectures or library builds because floating-point operations need not be bitwise identical. Compare the recorded configuration, runtime, checkpoint hashes and episode seeds when reproducing the run on another computer.
