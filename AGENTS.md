# AGENTS.md

## Purpose and scope

This repository is research code for the paper **“Hybrid Reinforcement Learning
Control for UAV Territory Guarding with Cooperative Angle-of-Arrival
Localization.”** It implements a two-stage, two-dimensional Multi-Agent Particle
Environment (MPE) experiment for cooperative counter-UAS research.

Treat this code as a reproducible research prototype, not as flight-ready control
software. It does not currently contain Isaac Sim/Isaac Lab integration, rigid-body
quadrotor dynamics, a low-level flight controller, hardware interfaces, or safety
assurance. Do not imply that a policy trained here can be deployed on an aircraft.

These instructions apply to the whole repository.

## Start here

Before changing code, read:

1. `README.md` for the supported workflows.
2. `docs/requirements.md` and `requirements.txt` for the Python environment.
3. The relevant entry point in `scripts/train/` or `scripts/render/`.
4. The matching scenario in `envs/mpe/scenarios/`.
5. The matching runner in `runner/bifamily/`.

The undergraduate-oriented background and paper list are in
`docs/undergraduate_reading_guide.md`. A guided PyTorch lesson is in
`docs/pytorch_coding_tutorial.md`, with runnable companion code at
`scripts/tutorials/pytorch_actor_critic_basics.py`.

## Repository architecture

- `config.py`: shared command-line arguments and PPO/RNN hyperparameters.
- `envs/mpe/`: MPE physics, Gym-style spaces, and scenario callbacks.
- `envs/mpe/scenarios/protected_zone_stage1.py`: ground-truth-state curriculum.
- `envs/mpe/scenarios/protected_zone_stage2.py`: bearing-only CPF belief state,
  threat calculation, capture/breach events, observations, and base rewards.
- `boa/cpf.py`: constant-acceleration particle filter and AoA likelihood model.
- `boa/threat.py`: covariance look-ahead and protected-zone collision time.
- `algorithms/r_mappo/`: the PPO/MAPPO optimization and recurrent actor-critic.
- `algorithms/utils/act.py`: continuous, discrete, and tuple action heads.
- `runner/bifamily/mpe_runner_bifamily.py`: Stage-1 defender/intruder rollout,
  training, metrics, and checkpoints.
- `runner/bifamily/mpe_runner_boa_hybrid.py`: Stage-2 hybrid-action rollout,
  learned assignment transfer, CPF reward augmentation, metrics, and checkpoints.
- `utils/shared_buffer.py`: on-policy trajectory storage and GAE returns.
- `scripts/train_stage{1,2}_bifamily.sh`: canonical experiment launchers.
- `scripts/render/`: deterministic checkpoint evaluation and trajectory export.
- `scripts/plot_algorithm_comparison.py`: TensorBoard/JSON comparison plots.
- `results/`, `render_output/`, `plots/`, `paper_figs/`: generated or example
  artifacts; do not hand-edit them to make an experiment appear better.

## Conceptual model

- Defenders protect a circular zone; intruders attempt to enter it.
- Stage 1 exposes ground-truth geometry and teaches basic pursuit, assignment,
  formation, and interception behavior.
- Stage 2 replaces defender knowledge of intruders with one centralized particle
  filter per intruder. Assigned defenders supply noisy bearings.
- The defender action is hybrid: a two-dimensional continuous velocity command
  plus a discrete target assignment (`0` means unassigned and `1..N` maps to
  intruder `0..N-1` in the runners).
- Intruders use continuous two-dimensional velocity actions.
- Training is centralized-training/decentralized-execution style. Defender and
  intruder families have separate shared policies and replay buffers.
- `rmappo` enables recurrent policies, `mappo` disables recurrence, and `ippo`
  disables the centralized critic. In these stage-specific entry points, do not
  assume that every algorithm listed by `config.py` is supported.
- Checkpoints are `actor.pt`, `critic.pt`, `intruder_actor.pt`, and
  `intruder_critic.pt`. Stage 2 must start from a compatible Stage-1 checkpoint.

## Important invariants

Preserve these unless the task explicitly changes the experiment definition:

- `num_agents == num_defenders + num_intruders`.
- Agent ordering is defenders first, then intruders.
- All agents receive fixed-length observations. If an observation layout changes,
  update both stages, spaces, runners, renderers, checkpoints/documentation, and
  add an explicit shape check.
- Stage-1 and Stage-2 policy shapes must remain checkpoint-compatible when Stage-1
  transfer is expected.
- The continuous environment action is a 2-vector. The defender policy's tuple
  action must be split before `env.step`; assignment is passed separately through
  the vector environment into `world._pz_assignments`.
- Captured intruders stay inactive, do not generate repeat capture rewards, and are
  masked consistently in rewards, observations, assignments, and metrics.
- Angle residuals must be wrapped across `-pi/pi`; covariance matrices and particle
  weights must remain finite; normalized weights must sum to one.
- Terminal conditions are breach, capture of all intruders, or the time limit.
- Keep the physical time step (`world.dt`) distinct from the episode step count.
- Defense/attack success, rewards, CPF uncertainty, and threat metrics must reflect
  environment outcomes; never infer scientific performance from reward alone.

## Configuration caveats

- Several flags in `config.py` use `action='store_false'`; passing the flag disables
  a feature even when its name begins with `--use_`. Read the declaration rather
  than guessing its meaning.
- `train_protected_zone_stage1_bifamily.py` forces `episode_length = 300`.
- `train_protected_zone_stage2_bifamily.py` currently forces 10M environment steps,
  episode length 300, 128 rollout processes, learning rates `3e-4`, and four
  minibatches after argument parsing. A command-line value for one of these is not
  effective until that override is removed. If editing configuration, establish a
  single source of truth and update the README and shell wrappers.
- Stage 2 can auto-select the newest Stage-1 model, while the canonical shell script
  supplies an exact directory. Prefer an explicit `--model_dir` for reproducible
  experiments.
- CUDA, Apple MPS, and CPU are selected in `utils/device_utils.py`. Keep tensors in
  `float32` unless a reviewed numerical change requires otherwise.

## Development workflow

Create the documented Python 3.10 environment and install dependencies:

```bash
conda create -n ha-rmappo python=3.10
conda activate ha-rmappo
pip install -r requirements.txt
```

Canonical full runs:

```bash
bash scripts/train_stage1_bifamily.sh <run-id>
bash scripts/train_stage2_bifamily.sh <run-id>
bash scripts/render_stage1.sh
bash scripts/render_stage2.sh
tensorboard --logdir results/MPE --port 6006
```

Do not launch a 10M-step training job merely to validate a code edit. First perform
fast checks. At minimum:

```bash
python -m compileall -q algorithms boa envs runner scripts utils config.py
python scripts/train/train_protected_zone_stage1_bifamily.py \
  --algorithm_name mappo --experiment_name smoke \
  --num_agents 3 --num_defenders 2 --num_intruders 1 \
  --n_rollout_threads 1 --n_training_threads 1 --num_env_steps 300 --cuda
```

The Stage-1 entry point fixes the episode length at 300, so 300 environment steps
produce one update. `--cuda` disables CUDA because of the `store_false` declaration.
Use a unique experiment name. A Stage-2 smoke test should be added/refactored rather
than run through the current entry point, because that file overwrites small CLI
limits.

There is no dedicated automated test suite yet. For nontrivial changes, add focused
`pytest` tests (preferably under `tests/`) for pure calculations and environment
contracts. Useful first targets are:

- angle wrapping, AoA variance, particle-weight normalization, ESS, and resampling;
- covariance symmetry/finite values and threat-time edge cases;
- reset/step observation and action shapes for 2v1, 5v2, and 7v3;
- capture/breach termination and one-time event accounting;
- tuple-action packing/unpacking and assignment indexing;
- checkpoint save/load compatibility.

## Change discipline

- Make the smallest change that answers the research question.
- Keep environment dynamics, state estimation, reward shaping, and learning logic
  separated. A change to one should not silently redefine another.
- Do not silently change defaults, reward coefficients, observation features,
  termination rules, or metric definitions. Document the change and explain why old
  checkpoints/results are or are not comparable.
- Avoid hard-coded experiment names and machine-specific paths.
- Use seeded runs, but do not claim reproducibility from a single seed. Record the
  seed, commit, command/effective config, device, package versions, and checkpoint.
- Compare algorithms with matched environment steps, seeds, scenarios, and evaluation
  episodes. Report mean and dispersion across seeds, success/breach rates, capture
  time, and safety violations—not just the best return curve.
- Never tune on the final test scenarios. Separate training, validation, and held-out
  scenario distributions.
- Keep generated models, event files, videos, and bulk plots out of source commits
  unless the task explicitly updates release artifacts.
- Preserve unrelated user changes in a dirty worktree.

## Mentoring and research expectations

For undergraduate contributions, prefer a sequence that can be verified in days:

1. Reproduce one short 2v1 Stage-1 run and explain every observation/action field.
2. Trace PPO data from environment step to buffer, GAE, clipped loss, and update.
3. Validate CPF behavior independently with synthetic trajectories.
4. Reproduce a small Stage-2 baseline with at least three seeds.
5. Change one hypothesis at a time and write an ablation plan before a long run.
6. Only then port the task to Isaac Lab, first matching the 2D kinematic baseline,
   then adding 3D dynamics, sensors, delays, domain randomization, and safety limits.

Every experiment note or pull request should answer: What changed? What hypothesis
does it test? What is the baseline? Which seeds and effective settings were used?
What failed? Are old checkpoints and plots still comparable?

## Isaac Sim / Isaac Lab migration boundary

Use Isaac Lab for a future physics-based implementation; Isaac Sim supplies the
simulation platform, while Isaac Lab supplies vectorized robot-learning task and RL
workflows. Keep that migration in a separate package or clearly isolated subtree.
Do not add Isaac dependencies to this lightweight MPE environment without an
explicit migration task.

Before comparing an Isaac result with this repository, match and test coordinate
frames, units, action semantics, control frequency, termination, initial-state
distribution, observation availability, and metrics. A sensible migration order is:

1. scripted zero/random controllers and deterministic reset tests;
2. one vehicle with velocity-command tracking;
3. 2v1 full-state pursuit with the same normalized observations;
4. multi-agent vectorization and MAPPO/IPPO baselines;
5. bearing sensor and CPF integration;
6. latency/noise/dropout, collision avoidance, actuator limits, and domain
   randomization;
7. held-out evaluation and only then hardware-in-the-loop work.

Keep learned high-level guidance behind a conventional safety and low-level control
layer. Any real-world interception or mitigation work requires institutional safety,
legal, airspace, and research-ethics review.
