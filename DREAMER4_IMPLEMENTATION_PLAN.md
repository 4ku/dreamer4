# Dreamer 4 PyTorch Implementation Plan

## Status

| Module | Status |
|--------|--------|
| Module 1: Core Transformer Building Blocks | pending |
| Module 2: Causal Tokenizer | pending |
| Module 3: Dynamics + Shortcut Forcing | pending |
| Module 4: Agent Heads (policy, reward, value) | pending |
| Module 5: Imagination Training | pending |
| Module 6: Environment Interface + Online Loop | pending |
| Module 7: Config, Logging, Checkpointing | pending |
| End-to-end Integration Test | pending |

## Architecture Overview

Dreamer 4 has three training phases and four major components:

**Phase 1: World Model Pretraining** — Train causal tokenizer on videos, then train dynamics model on tokenized videos + actions using shortcut forcing.

**Phase 2: Agent Finetuning** — Insert agent tokens into dynamics transformer. Train policy head (behavior cloning) and reward head conditioned on tasks.

**Phase 3: Imagination Training** — Finetune policy via PMPO reinforcement learning on rollouts generated inside the world model. Train value head with TD(lambda).

The **online training loop** (DreamerV3-style) wraps all phases: collect experience → train world model → imagine rollouts → update policy.

---

## Component Breakdown

### 1. Efficient Block-Causal Transformer (foundation for everything)

Shared backbone used by both the tokenizer and dynamics model.

**Key elements:**
- Pre-layer RMSNorm, RoPE positional encoding, SwiGLU MLP
- QKNorm + attention logit soft capping (stability)
- Modality-aware space attention (different masks for encoder/decoder/WM modes)
- Causal time attention (every 4th layer by default)
- GQA (grouped query attention) for dynamics model KV cache efficiency
- Factored space-only + time-only attention layers

**Differences from existing `dreamer4/dreamer4/model.py`:**
- Existing code uses sinusoidal positions; paper uses RoPE
- Existing code lacks QKNorm, attention logit soft capping, GQA
- Existing code has a single `MultiheadSelfAttention` without GQA support

### 2. Causal Tokenizer

Compresses video frames to continuous latent representations.

**Architecture:** Encoder + Decoder, both using the block-causal transformer.
- Encoder: patchify images (16x16 patches in paper, 4x4 in existing code) → MAE masking → transformer → linear projection to bottleneck dim → tanh
- Decoder: project bottleneck back up → concat with learned patch queries → transformer → reconstruct patches
- Loss: MSE + 0.2 * LPIPS (with RMS loss normalization)
- MAE dropout: p ~ U(0, 0.9) per image

**For robotics setup:** 224x224 images with 16x16 patches = 196 spatial tokens per camera. Three cameras means 588 patch tokens total (or three separate tokenizer streams).

### 3. Dynamics Model (Shortcut Forcing)

Predicts future latent representations given actions.

**Architecture:** Operates on interleaved sequence of: actions, shortcut signals (tau, d), spatial tokens (packed from tokenizer), register tokens, (optional) agent tokens.

**Shortcut forcing objective** (the core training objective):
- Based on flow matching + shortcut models + diffusion forcing
- X-prediction (not V-prediction) to prevent error accumulation
- Ramp loss weight: w(tau) = 0.9*tau + 0.1
- Bootstrap loss distills two half-steps into one full step
- At inference: K=4 sampling steps per frame, context corruption tau_ctx=0.1

**Key formulas from paper (Section 3.2, Eq. 7):**
- Flow term (d=d_min): `||f(z_tilde, tau, d, a) - z1||^2`
- Bootstrap term (d>d_min): `(1-tau)^2 * ||v_hat - sg(b'+b'')/2||^2`
- Where b' and b'' are velocity predictions from two half-steps

### 4. Agent (Policy, Reward, Value Heads)

**Phase 2 — Behavior cloning + reward model:**
- Agent tokens inserted into dynamics transformer (isolated attention: agent tokens see all modalities, but no other modality sees agent tokens)
- Task embeddings as input to agent tokens
- Policy head: MLP with multi-token prediction (L=8)
- Reward head: MLP with symexp twohot output, also MTP

**Phase 3 — Imagination training (RL):**
- Value head: MLP with symexp twohot, trained with TD(lambda)
- Policy: PMPO objective (sign of advantages, no return normalization)
- KL regularization to behavioral prior (reverse KL, beta=0.3)
- Discount gamma=0.997, lambda for TD(lambda)
- Rollouts from dataset contexts, one rollout per context

### 5. Online Training Loop (DreamerV3-style)

The paper focuses on offline training. For online support (based on DreamerV3's `embodied/run/train.py`):

```
loop:
  1. Environment → Replay: collect transitions via policy
  2. Replay → World Model: sample batch, train tokenizer + dynamics
  3. World Model → Agent: imagine rollouts from replay contexts
  4. Agent: train policy via PMPO + value via TD(lambda)
```

Key difference from DreamerV3: the world model is a transformer (not RSSM), so "state" is the KV cache + last K frames of context, not a compact recurrent state.

### 6. Multi-Camera Support

For robotics use case (3 cameras, 224x224):
- Each camera gets its own tokenizer stream (or shared tokenizer with camera-ID modality)
- Tokenizer representations from all cameras are concatenated along the spatial dimension before feeding to dynamics
- The modality-aware attention system already supports this via `TokenLayout`

---

## Implementation Order

Build bottom-up, test each component in isolation with clear examples.

### Module 1: Core Transformer Building Blocks
- RMSNorm, RoPE, QKNorm, SwiGLU MLP
- MultiheadAttention with GQA support + logit soft capping
- SpaceAttention (modality-aware masks)
- TimeAttention (causal, every Nth layer)
- BlockCausalLayer, BlockCausalTransformer
- **Tests:** Shape checks, causal mask correctness, GQA equivalence, attention pattern visualization

### Module 2: Causal Tokenizer
- Patchify/unpatchify utilities
- MAE masking module
- Encoder (patches → latent bottleneck)
- Decoder (latent → reconstructed patches)
- Tokenizer (encoder + decoder)
- Training loop for tokenizer (MSE + LPIPS)
- **Tests:** Reconstruction on toy images, bottleneck compression ratio, MAE masking stats

### Module 3: Shortcut Forcing / Dynamics
- Flow matching sampling (noise schedule, corruption)
- Shortcut model: step/signal conditioning
- Action encoder (continuous + discrete)
- Dynamics model (interleaved sequence processing)
- Shortcut forcing loss (flow term + bootstrap term + ramp weight)
- Sampling / generation (K-step denoising)
- **Tests:** Single-step denoising on toy data, multi-step generation, loss computation correctness

### Module 4: Agent Heads
- Symexp twohot distribution (for rewards and values)
- Policy head (continuous: Normal/TruncNormal, discrete: categorical)
- Reward head with multi-token prediction
- Value head with TD(lambda) targets
- PMPO objective
- **Tests:** Symexp roundtrip, PMPO gradient direction, lambda-return computation

### Module 5: Imagination Training
- Rollout generation inside world model
- Reward annotation from reward head
- Value estimation and lambda-return computation
- PMPO policy optimization
- Value head optimization
- **Tests:** Rollout shape checks, reward/value gradient flow

### Module 6: Environment Interface + Online Loop
- DMControl wrapper (image observations, continuous actions)
- Multi-camera environment wrapper
- Replay buffer (supports both offline and online)
- Driver (parallel env execution)
- Training loop: collect → train WM → imagine → train agent
- **Tests:** DMControl smoke test, replay buffer sampling, online loop integration

### Module 7: Config + Logging + Checkpointing
- YAML/dataclass config system
- Checkpoint save/load with resume support
- Distributed training utilities (DDP/FSDP)

---

## File Structure

```
dreamer4_pytorch/
  config/
    defaults.yaml
    dmcontrol.yaml
    robotics.yaml
  dreamer4/
    __init__.py
    transformer.py        # Module 1: block-causal transformer
    tokenizer.py          # Module 2: causal tokenizer
    dynamics.py           # Module 3: dynamics + shortcut forcing
    agent.py              # Module 4+5: policy, reward, value, imagination
    distributions.py      # symexp twohot, truncated normal, etc.
    objectives.py         # PMPO, shortcut forcing loss, etc.
    utils.py              # patchify, RMS normalization, etc.
  envs/
    __init__.py
    dmcontrol.py          # DMControl wrapper
    wrappers.py           # action normalization, multi-camera, etc.
  training/
    __init__.py
    replay.py             # replay buffer
    driver.py             # parallel env driver
    train.py              # main training loop (online + offline)
    train_tokenizer.py    # standalone tokenizer pretraining
  tests/
    test_transformer.py
    test_tokenizer.py
    test_dynamics.py
    test_agent.py
    test_distributions.py
    test_objectives.py
    test_envs.py
    test_integration.py
  examples/
    tokenizer_demo.py     # train tokenizer on toy video
    dynamics_demo.py      # train dynamics on toy sequences
    dmcontrol_online.py   # full online training on DMControl
  README.md
  requirements.txt
  setup.py
```

---

## Key Design Decisions

1. **RoPE vs sinusoidal**: Use RoPE as in the paper (the existing code uses sinusoidal positions)
2. **GQA**: Implement from the start; the paper uses it for dynamics model efficiency
3. **Patch size**: Default 16x16 as in paper; configurable for smaller images (4x4 for 64x64)
4. **Multi-camera**: Treat each camera as a separate modality in `TokenLayout`; share tokenizer weights
5. **Online + Offline**: Unified training loop that supports both modes; offline = no environment driver
6. **Action space**: Support both continuous (robotics/DMControl) and discrete (future Minecraft)
7. **Start small**: Begin with DMControl at 64x64 or 128x128 for fast iteration, scale to 224x224 later

---

## Reference

- Paper: https://arxiv.org/pdf/2509.24527
- Existing unofficial impl: `dreamer4/dreamer4/` (PyTorch, partial)
- DreamerV3 official impl: `dreamerv3/` (PyTorch, online training loop reference)
