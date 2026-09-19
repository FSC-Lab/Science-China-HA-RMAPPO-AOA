# Undergraduate Reading Guide: RL, PPO, Isaac Lab, and Counter-UAS

This guide is ordered for a student starting from this repository. The goal is to
understand the current 2D experiment before attempting a physics-based Isaac Lab
port. The links favor original papers, official documentation, and open versions.

## A practical 14-week path

### Weeks 1-2: reinforcement-learning foundations

Learn Markov decision processes, policies, value functions, Bellman equations,
policy gradients, actor-critic methods, discounting, and the difference between
on-policy and off-policy data.

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Read Chapters
  1-3, 6, and 13: <http://incompleteideas.net/book/the-book-2nd.html>.
- David Silver's RL lectures are a useful second explanation:
  <https://www.davidsilver.uk/teaching/>.
- Implement tabular value iteration and a small REINFORCE/actor-critic example before
  changing this repository.

### Weeks 3-4: basic PyTorch and neural-network coding

Work through the official [PyTorch beginner
series](https://docs.pytorch.org/tutorials/beginner/basics/) in this order:
tensors, datasets/data loaders, model construction, autograd, optimization, and
saving/loading. Be able to explain:

- tensor shape, rank, `dtype`, and `device`;
- broadcasting, indexing, reshaping, concatenation, and batch dimensions;
- `requires_grad`, the computation graph, `.backward()`, `.grad`, `.detach()`, and
  `torch.no_grad()`;
- `nn.Module`, `forward`, registered parameters, `state_dict`, `train()`, and
  `eval()`;
- losses, `optimizer.zero_grad()`, backpropagation, gradient clipping, and
  `optimizer.step()`;
- `Categorical` and `Normal` distributions, action sampling, `log_prob`, and
  entropy;
- NumPy/tensor conversion and safe CPU/GPU/MPS movement.

Complete [the project-specific PyTorch coding
tutorial](pytorch_coding_tutorial.md) and run
`scripts/tutorials/pytorch_actor_critic_basics.py`. Do not proceed until you can
predict the shape of every intermediate tensor and explain why policy-gradient code
uses log probabilities rather than raw probabilities.

### Weeks 5-6: PPO and the implementation in this repository

Read the PPO and GAE papers, then trace one batch through:

`scenario -> env.step -> BiFamilyRunner -> SharedReplayBuffer -> compute_returns ->
R_MAPPO.ppo_update`.

Pay special attention to the probability ratio, clipped surrogate, entropy bonus,
value loss, GAE, masks at termination, advantage normalization, and gradient
clipping.

- OpenAI Spinning Up PPO explanation:
  <https://spinningup.openai.com/en/latest/algorithms/ppo.html>.
- Schulman et al., “Proximal Policy Optimization Algorithms”:
  <https://arxiv.org/abs/1707.06347>.
- Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage
  Estimation”: <https://arxiv.org/abs/1506.02438>.

### Weeks 7-8: multi-agent PPO and partial observability

Learn centralized training with decentralized execution (CTDE), parameter sharing,
centralized versus local critics, recurrent policies, non-stationarity, and
cooperative-competitive evaluation.

- Yu et al., “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games”
  (MAPPO/IPPO): <https://arxiv.org/abs/2103.01955>.
- Kuba et al., “Trust Region Policy Optimisation in Multi-Agent Reinforcement
  Learning” (HATRPO/HAPPO): <https://openreview.net/forum?id=EcGGFkNTxdJ>.
- Lowe et al., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
  Environments”: <https://arxiv.org/abs/1706.02275>.

Use 2v1 first. Draw the exact local observation, centralized critic input, recurrent
state, continuous action, assignment action, reward, and terminal mask. Confirm each
against the code rather than relying on names in the paper or README.

### Weeks 9-10: estimation and the Stage-2 sensing loop

Review Bayes filters, sequential importance sampling, effective sample size,
resampling, bearing-only observability, covariance ellipses, and numerical stability.
Then write small deterministic tests for `boa/cpf.py` and `boa/threat.py`.

A useful check is a stationary target observed by two well-separated sensors: the
posterior position uncertainty should usually contract over repeated informative
bearings. Also test collinear/near-collinear geometry and angle wraparound.

### Weeks 11-12: Isaac Sim and Isaac Lab

Isaac Sim and Isaac Lab are related but not interchangeable. Isaac Sim is the
robotics simulation platform; Isaac Lab is the robot-learning framework and task
workflow built on it. For new RL task development, start from current Isaac Lab
documentation rather than old standalone Isaac Gym tutorials.

- Isaac Lab quick start: <https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html>.
- Create a task and train an RL agent:
  <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html>.
- RL workflow and supported libraries:
  <https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html>.
- Training/debugging guide:
  <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html>.
- Makoviychuk et al., “Isaac Gym: High Performance GPU-Based Physics Simulation for
  Robot Learning”: <https://arxiv.org/abs/2108.10470>.
- Mittal et al., “Orbit: A Unified Simulation Framework for Interactive Robot
  Learning Environments” (the predecessor of Isaac Lab):
  <https://arxiv.org/abs/2301.04195>.

First complete Cartpole and one direct-workflow task. Then build a minimal quadrotor
velocity-tracking task and validate zero/random/scripted agents before training PPO.

### Weeks 13-14: a scoped research reproduction

Reproduce one 2v1 result with at least three seeds. Pre-register the command,
environment steps, evaluation episodes, seed set, primary metric, and failure
criteria. Report success and breach rates, time-to-capture, collision/safety counts,
return, and wall-clock/sample cost with uncertainty across seeds.

For a first extension, choose only one of: delayed bearings, missed detections,
assignment constraints, intruder-policy generalization, or transfer from planar MPE
to a matched Isaac Lab task. Do not change sensing, dynamics, rewards, and algorithm
simultaneously.

## Fifteen-paper counter-UAS reading list

Read 1-5 for the system landscape, 6-9 for sensing/localization, and 10-15 for the
territory-guarding and cooperative-interception thread closest to this repository.

1. J. Wang, Y. Liu, and H. Song, “Counter-Unmanned Aircraft System(s) (C-UAS):
   State of the Art, Challenges and Future Trends,” *IEEE Aerospace and Electronic
   Systems Magazine*, 2021. <https://doi.org/10.1109/MAES.2020.3015537>
2. S. Park, H. T. Kim, S. Lee, H. Joo, and H. Kim, “Survey on Anti-Drone Systems:
   Components, Designs, and Challenges,” *IEEE Access*, 2021.
   <https://doi.org/10.1109/ACCESS.2021.3065926>
3. V. U. Castrillo, A. Manco, D. Pascarella, and G. Gigante, “A Review of
   Counter-UAS Technologies for Cooperative Defensive Teams of Drones,” *Drones*,
   2022. <https://doi.org/10.3390/drones6030065>
4. X. Shi, C. Yang, W. Xie, C. Liang, Z. Shi, and J. Chen, “Anti-Drone System with
   Multiple Surveillance Technologies: Architecture, Implementation, and
   Challenges,” *IEEE Communications Magazine*, 2018.
   <https://doi.org/10.1109/MCOM.2018.1700430>
5. V. Chamola, P. Kotesh, A. Agarwal, Naren, N. Gupta, and M. Guizani, “A
   Comprehensive Review of Unmanned Aerial Vehicle Attacks and Neutralization
   Techniques,” *Ad Hoc Networks*, 2021.
   <https://doi.org/10.1016/j.adhoc.2020.102324>
6. M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-Based Drone
   Detection and Identification Using Deep Learning Approaches,” *Future Generation
   Computer Systems*, 2019. <https://doi.org/10.1016/j.future.2019.05.007>
7. M. S. Allahham et al., “DroneRF Dataset: A Dataset of Drones for RF-Based
   Detection, Classification and Identification,” *Data in Brief*, 2019.
   <https://doi.org/10.1016/j.dib.2019.104313>
8. A. Coluccia et al., “Drone-vs-Bird Detection Challenge at IEEE AVSS2017,”
   *14th IEEE AVSS*, 2017. <https://doi.org/10.1109/AVSS.2017.8078464>
9. J. Xu, M. Ma, and C. L. Law, “Cooperative Angle-of-Arrival Position
   Localization,” *Measurement*, 2015.
   <https://doi.org/10.1016/j.measurement.2014.09.023>
10. D. Mao, Y. Fang, and X. Gao, “Target Tracking Method with Bearings-Only
    Measurements Based on Reinforcement Learning,” *IEICE Communications Express*,
    2016. <https://doi.org/10.1587/comex.2015XBL0154>
11. C. V. Analikwu and H. M. Schwartz, “Reinforcement Learning in the Guarding a
    Territory Game,” *IEEE FUZZ-IEEE*, 2016.
    <https://doi.org/10.1109/FUZZ-IEEE.2016.7737798>
12. W. Du, T. Guo, J. Chen, B. Li, G. Zhu, and X. Cao, “Cooperative Pursuit of
    Unauthorized UAVs in Urban Airspace via Multi-Agent Reinforcement Learning,”
    *Transportation Research Part C*, 2021.
    <https://doi.org/10.1016/j.trc.2021.103122>
13. C. de Souza, R. Newbury, A. Cosgun, P. Castillo, B. Vidolov, and D. Kulic,
    “Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning,” *IEEE
    Robotics and Automation Letters*, 2021.
    <https://doi.org/10.1109/LRA.2021.3068952>
14. R. Zhang, Q. Zong, X. Zhang, L. Dou, and B. Tian, “Game of Drones: Multi-UAV
    Pursuit-Evasion Game with Online Motion Planning by Deep Reinforcement Learning,”
    *IEEE Transactions on Neural Networks and Learning Systems*, 2023.
    <https://doi.org/10.1109/TNNLS.2022.3146976>
15. W. Zhuang, L. Qian, and H. H.-T. Liu, “Hybrid Reinforcement Learning Control for
    UAV Territory Guarding with Cooperative Angle-of-Arrival Localization,”
    *Science China Technological Sciences*, 2026.
    <https://doi.org/10.1007/s11431-025-3331-4>

## Questions to answer while reading

For every paper, write a half-page note:

- What precise threat model and protected asset are assumed?
- What information is available to each defender at execution time?
- Are detection, localization, tracking, assignment, guidance, and mitigation modeled
  separately or collapsed into one policy?
- What aircraft dynamics, sensor errors, delays, communication limits, and safety
  constraints are omitted?
- What is the strongest baseline, and are compute and observations matched?
- Are results averaged across seeds and held-out scenarios?
- Which assumption would fail first in a real C-UAS deployment?

Keep the project focused on lawful detection, tracking, safe interception research,
and robust evaluation. Physical mitigation, RF interference, or real flight testing
must not proceed without appropriate legal, spectrum, airspace, institutional, and
safety authorization.
