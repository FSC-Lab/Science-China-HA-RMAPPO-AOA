# PyTorch Coding Tutorial for This Project

This tutorial teaches the minimum PyTorch needed to read and modify the HA-RMAPPO
code. It assumes basic Python and NumPy, but no previous deep-learning framework.

## 1. Setup and official references

Install the repository dependencies in the environment described by `README.md`:

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

Use the official tutorials as the primary reference:

- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/)
- [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Build a model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

Run the companion program:

```bash
python scripts/tutorials/pytorch_actor_critic_basics.py
```

It performs a few PPO-style updates on a synthetic batch. It is deliberately small:
it teaches tensor and loss mechanics, not a complete RL experiment.

## 2. Tensors: values plus shape, type, and device

```python
import numpy as np
import torch

obs_np = np.zeros((32, 5, 20), dtype=np.float32)
obs = torch.from_numpy(obs_np)

print(obs.shape)   # [rollout environments, agents, observation features]
print(obs.dtype)   # torch.float32
print(obs.device)  # cpu initially

flat_obs = obs.reshape(32 * 5, 20)
assert flat_obs.shape == (160, 20)
```

RL code often has several leading dimensions: time `T`, parallel environments `E`,
agents `A`, and features `F`. Write those symbols beside a complicated expression
before reshaping it. A reshape that preserves element count can still scramble the
meaning of axes.

Common operations:

```python
x = torch.randn(8, 4)
first_feature = x[:, 0]                 # [8]
first_feature_column = x[:, 0:1]        # [8, 1]
joined = torch.cat([x, x], dim=-1)      # [8, 8]
stacked = torch.stack([x, x], dim=0)    # [2, 8, 4]
mean_per_batch = x.mean(dim=-1)         # [8]
```

`cat` extends an existing axis; `stack` creates a new axis. Preserving the last
singleton dimension (`[B, 1]`) often prevents accidental broadcasting in value-loss
code.

### Exercise

Given an observation tensor `[T, E, A, F] = [300, 32, 3, 20]`, reshape it for one
MLP call and then restore it. Assert equality with the original tensor. The expected
MLP input shape is `[28_800, 20]`.

## 3. Devices and NumPy conversion

All tensors in an operation must normally share a device. This project chooses CPU,
CUDA, or MPS in `utils/device_utils.py`.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(64, 20, dtype=torch.float32, device=device)

# Convert safely for plotting or NumPy-only code.
x_np = x.detach().cpu().numpy()
```

Do not call `.numpy()` directly on a CUDA/MPS tensor or on a tensor still attached to
an autograd graph. Avoid repeatedly moving data between CPU and GPU inside a training
loop.

## 4. Models with `nn.Module`

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MLP(20, 2).to(device)
y = model(torch.randn(32, 20, device=device))
assert y.shape == (32, 2)
```

Calling `model(x)` invokes `forward`. Submodules assigned as attributes are
registered automatically, so their parameters appear in `model.parameters()` and
`model.state_dict()`.

In this repository, start with:

- `algorithms/r_mappo/algorithm/r_actor_critic.py` for actor and critic modules;
- `algorithms/utils/mlp.py` and `algorithms/utils/rnn.py` for feature/RNN layers;
- `algorithms/utils/act.py` for action-distribution heads;
- `algorithms/r_mappo/algorithm/rMAPPOPolicy.py` for the optimizer-facing wrapper.

## 5. Autograd and one optimizer step

```python
model = MLP(4, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

x = torch.randn(16, 4, device=device)
target = torch.randn(16, 1, device=device)

prediction = model(x)
loss = torch.nn.functional.mse_loss(prediction, target)

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
optimizer.step()
```

The order matters:

1. forward pass builds the computation graph;
2. loss reduces predictions to a scalar objective;
3. `zero_grad` clears gradients accumulated by earlier backward calls;
4. `backward` populates parameter `.grad` tensors;
5. optional clipping bounds the total gradient norm;
6. `step` changes the parameters.

PyTorch accumulates gradients by default. Forgetting `zero_grad` unintentionally sums
gradients across batches. Conversely, `detach()` intentionally breaks gradient flow;
using it on the current policy's loss would stop learning.

### Exercise

Print the loss before and after 100 updates on a fixed regression batch. Then comment
out `optimizer.step()` and explain why the loss stops improving.

## 6. Action distributions for RL

A policy produces distribution parameters, samples an action, and records its log
probability. PPO later compares the current log probability with the stored old one.

### Discrete assignment

```python
from torch.distributions import Categorical

logits = torch.randn(32, 3, device=device)  # unassigned, target 0, target 1
dist = Categorical(logits=logits)
action = dist.sample()                      # [32]
log_prob = dist.log_prob(action)            # [32]
entropy = dist.entropy()                    # [32]
```

### Continuous velocity

```python
from torch.distributions import Normal

mean = torch.zeros(32, 2, device=device)
log_std = torch.zeros(2, device=device)
dist = Normal(mean, log_std.exp())
action = dist.sample()                       # [32, 2]
log_prob = dist.log_prob(action).sum(-1)     # joint log-probability, [32]
entropy = dist.entropy().sum(-1)             # [32]
```

Independent action dimensions have additive log probabilities, which is why the
last dimension is summed. A hybrid policy adds the continuous and discrete
log-probabilities for the joint action. Inspect `algorithms/utils/act.py` and verify
how this repository packs tuple actions.

## 7. Actor-critic and PPO loss

The actor selects actions. The critic estimates expected discounted return. For an
observation batch `obs` and centralized critic batch `share_obs`:

```python
action_dist = actor(obs)
action = action_dist.sample()
new_log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
value = critic(share_obs)
```

For stored old log probabilities and normalized advantages:

```python
ratio = torch.exp(new_log_prob - old_log_prob)
unclipped = ratio * advantage
clipped = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
policy_loss = -torch.minimum(unclipped, clipped).mean()

value_loss = 0.5 * (returns - value).square().mean()
loss = policy_loss + value_loss - 0.01 * entropy.mean()
```

Interpretation:

- `ratio == 1` means the action is equally likely under old and current policies;
- clipping discourages a single batch from changing the policy too far;
- positive advantage increases the probability of an action, while negative
  advantage decreases it;
- the entropy term discourages premature collapse to a nearly deterministic policy;
- the critic learns returns but does not directly choose the action.

Compare this simplified version line-by-line with `R_MAPPO.ppo_update` in
`algorithms/r_mappo/r_mappo.py`.

### Exercise

Construct four samples with ratios `[0.5, 0.9, 1.1, 1.5]` and positive advantages.
Compute the unclipped and clipped objectives by hand for `clip_param=0.2`, then
confirm with PyTorch. Repeat with negative advantages and explain why the minimum is
important.

## 8. Recurrent policies and masks

RMAPPO adds a recurrent hidden state. Conceptually:

```python
output, next_hidden = gru(features, hidden)
hidden = next_hidden * mask
```

`mask` is zero at an episode boundary and one during a continuing trajectory. This
prevents information from the previous episode leaking into a new one. Track the
expected shape of hidden states carefully: this repository carries rollout-thread,
agent, recurrent-layer, and hidden-feature dimensions through buffers and flattens
them for network calls.

### Exercise

Run a GRU for three steps, set one environment's mask to zero before step two, and
verify that its reset output differs from the no-reset case while the other
environment is unchanged.

## 9. Training versus evaluation

```python
model.train()  # training behavior for dropout/batch normalization

model.eval()
with torch.no_grad():
    prediction = model(x)
```

`eval()` does not disable autograd, and `no_grad()` does not switch module behavior;
use both for inference. This repository calls `prep_training()` and `prep_rollout()`
to switch modes.

## 10. Checkpoints

Prefer state dictionaries:

```python
torch.save(model.state_dict(), "actor.pt")

restored = MLP(20, 2).to(device)
state = torch.load("actor.pt", map_location=device, weights_only=True)
restored.load_state_dict(state)
restored.eval()
```

The architecture and tensor shapes must match the saved state. This is why changing
observation length, hidden size, recurrence, or action layout can invalidate this
project's Stage-1 checkpoints for Stage 2.

For resuming training exactly, also save optimizer state, scheduler state, update
count, configuration, and random-number-generator states. The repository's current
checkpoints primarily save model weights.

## 11. Debugging checklist

When a training run fails, check in this order:

1. Print/assert tensor shapes at the environment, buffer, and policy boundaries.
2. Check `dtype` and `device` for every tensor in the failing expression.
3. Check observations, actions, rewards, log probabilities, values, advantages,
   losses, and gradients with `torch.isfinite`.
4. Confirm terminal masks reset RNN state and bootstrap logic correctly.
5. Confirm action bounds and discrete assignment indices.
6. Overfit one tiny fixed batch; failure usually indicates a loss/gradient bug.
7. Run one environment and one short update before parallel training.
8. Compare requested CLI values with the effective configuration printed by code.

Useful assertions:

```python
assert obs.ndim == 2
assert torch.isfinite(obs).all()
assert torch.isfinite(loss)
assert action.shape[-1] == 2
assert (probabilities >= 0).all()
assert torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))
```

## 12. Completion task

The student is ready to modify the PPO implementation after they can:

1. run the companion script and annotate every tensor shape;
2. implement the same actor and critic without copying;
3. explain the effect of removing `zero_grad`, `detach`, clipping, and entropy;
4. save and reload the actor and reproduce identical deterministic output;
5. trace one real repository rollout from NumPy observation through action sampling,
   buffer insertion, GAE, minibatch generation, loss, and optimizer step.
