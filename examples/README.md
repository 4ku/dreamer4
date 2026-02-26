# Toy Examples for the Block-Causal Transformer

Six runnable scripts that verify the transformer, tokenizer, and dynamics
model work end-to-end on synthetic tasks.  Each script trains a tiny
model, asserts that the loss drops significantly, and saves visualisation
plots to the `outputs/` directory.

## Prerequisites

```bash
cd dreamer4_pytorch
uv pip install -e .
```

## Running the examples

All commands are run from the `dreamer4_pytorch/` directory.

### 1. Copy Task

```bash
python -m examples.copy_task
```

The model overfits a random target tensor — the simplest check that
gradients flow and the architecture is wired correctly.

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/copy_task_loss.png` | Training loss (log scale) vs step. Should drop smoothly. |
| `outputs/copy_task_comparison.png` | Side-by-side heatmaps of target, model output, and their difference for one (batch, time) slice. After training the difference should be near zero. |

*MSE loss drops smoothly from ~1.5 to ~0.02 over 500 steps on a log scale, confirming that gradients flow correctly through the architecture.*

![copy_task_loss](outputs/copy_task_loss.png)

*Three heatmaps for one (batch=0, t=0) slice: the target tensor (left), the model output (center), and their difference (right). The target and output are visually near-identical; the difference map shows only small residual errors (within +-0.4), concentrated in a few spatial tokens.*

![copy_task_comparison](outputs/copy_task_comparison.png)

### 2. Pattern Sequence Continuation

```bash
python -m examples.next_token_prediction
```

K distinct pattern vectors cycle in a fixed order (A-B-C-D-A-B-C-D-...).
The model must learn to predict which pattern comes next.  Unlike random
noise, this task has real temporal structure — the model needs **causal
time attention** to detect where it is in the cycle and output the
correct continuation.

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/pattern_seq_loss.png` | Training loss (log scale) vs step. |
| `outputs/pattern_seq_similarity.png` | Cosine similarity between each prediction and the K pattern templates, with a per-step correctness indicator.  After training the model should pick the right pattern at every step. |

*Loss falls steeply from ~3e-1 to ~3e-5 over 500 steps, indicating the model quickly learns the repeating A-B-C-D cycle.*

![pattern_seq_loss](outputs/pattern_seq_loss.png)

*Left: cosine-similarity heatmap between each prediction and the four pattern templates (A-D). Black boxes mark the correct pattern -- high similarity aligns perfectly with them at every time step. Right: per-step correctness is 100%, confirming the model has learned the full cycle.*

![pattern_seq_similarity](outputs/pattern_seq_similarity.png)

### 3. Sine Wave Prediction

```bash
python -m examples.sine_wave
```

Each spatial position carries a sine wave with a unique frequency and
phase.  The model must learn to predict the next value from the history.

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/sine_wave_loss.png` | Training loss (log scale) vs step. |
| `outputs/sine_wave_prediction.png` | Ground-truth vs predicted sine values for 4 spatial positions. After training the predicted points should closely follow the true wave. |

*Loss decreases from ~8e-1 to ~5e-4 over 500 steps with periodic spikes (likely from learning-rate warmup restarts or different frequency components locking in). The overall trend is strongly downward.*

![sine_wave_loss](outputs/sine_wave_loss.png)

*Four subplots, one per spatial position (s=0, 3, 7, 11), each with a different frequency/phase. Blue circles are ground-truth sine values; orange crosses are model predictions. The predicted points closely track the true wave at all positions, demonstrating the model has learned diverse temporal dynamics from causal context alone.*

![sine_wave_prediction](outputs/sine_wave_prediction.png)

### 4. Bouncing Ball (Image Pipeline)

```bash
python -m examples.bouncing_ball
```

Generates synthetic 32x32 RGB video of a bright ball bouncing inside a
box.  The model predicts the next frame from the history, exercising the
**full image pipeline**: `patchify -> project -> transformer -> project -> unpatchify`.

This is the closest example to what the real tokenizer will do — the model
must learn spatial structure (where is the ball?) and temporal dynamics
(which direction is it moving?) from raw pixels.

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/bouncing_ball_loss.png` | Training loss (log scale) vs step. |
| `outputs/bouncing_ball_frames.png` | Three rows: ground-truth target frames, model-predicted frames, and the absolute difference.  After training the predictions should closely match. |

*Training loss (blue) drops steadily from ~3e-1 to ~1.5e-4 over 3000 steps. Eval loss (orange) is noisier but follows a clear downward trend, reaching ~5e-4 by the end, showing the model generalises to unseen trajectories.*

![bouncing_ball_loss](outputs/bouncing_ball_loss.png)

*Four bouncing-ball trajectories (8 frames each). For each trajectory, the TRUE row shows the ground-truth frames and the PRED row (blue border) shows the model's next-frame predictions. The predicted ball positions and shapes closely match the ground truth across all trajectories, confirming the model has learned both spatial appearance and temporal motion from raw 32x32 RGB pixels.*

![bouncing_ball_frames](outputs/bouncing_ball_frames.png)

### 5. Tokenizer Reconstruction (Causal Tokenizer)

```bash
python -m examples.tokenizer_recon
```

Trains the full **Causal Tokenizer** (Encoder -> bottleneck -> Decoder)
to reconstruct bouncing-ball video frames using the paper's exact training
objective (Section 3.1, Eq. 5):

    L = L_MSE + 0.2 * L_LPIPS    (both RMS-normalized)

MAE masking with `p ~ U(0, 0.9)` drops random encoder input patches.
MSE is computed on masked patches only (`recon_loss_from_mae`), while
LPIPS is computed on composited full images (`lpips_on_mae_recon`).
Both loss terms are RMS-normalized before combining (`RMSLossNormalizer`).

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/tokenizer_recon_loss.png` | Combined loss, MSE, LPIPS, and eval MSE vs step (log scale). |
| `outputs/tokenizer_recon_frames.png` | Ground-truth frames vs tokenizer reconstructions for several eval trajectories. |
| `outputs/tokenizer_recon_mae_pipeline.png` | MAE pipeline: original frames, masked encoder input, and reconstructed output. |

*Four loss curves over 3000 steps. Combined RMS-normalized loss (blue) stabilises around 1.0. MSE on masked patches (orange, dashed) and LPIPS on composited images (green, dotted) both drop by several orders of magnitude. Eval full-image MSE (red) falls to ~2e-5, confirming strong reconstruction quality.*

![tokenizer_recon_loss](outputs/tokenizer_recon_loss.png)

*Four trajectories (8 frames each) comparing ground-truth (TRUE) and tokenizer-decoded (RECON, blue border) frames. The reconstructed ball positions, shapes, and brightness closely match the originals across all time steps, showing the Encoder-Decoder pipeline faithfully reconstructs the input video.*

![tokenizer_recon_frames](outputs/tokenizer_recon_frames.png)

*MAE pipeline visualisation with 34% of patches masked. Top row: original frames. Middle row: encoder input after MAE masking -- several patches are zeroed out, visibly removing parts of the ball. Bottom row: decoder reconstruction, which recovers the full ball despite the missing patches, demonstrating the tokenizer's robustness to partial observations.*

![tokenizer_recon_mae_pipeline](outputs/tokenizer_recon_mae_pipeline.png)

### 6. Dynamics Bouncing Ball (Full Pipeline)

```bash
python -m examples.dynamics_bouncing_ball
```

End-to-end test of the **Shortcut Forcing dynamics pipeline** with a
real causal tokenizer.  Runs three phases:

1. **Tokenizer pretraining** — trains the full Encoder-Decoder with
   the paper's Eq. 5 loss (MSE + 0.2·LPIPS, RMS-normalized) so the
   tokenizer learns a meaningful latent space.
2. **Dynamics training** — freezes the tokenizer, encodes all data
   into the latent space via `pack_bottleneck_to_spatial`, and trains
   the `DynamicsModel` with shortcut forcing (flow matching +
   bootstrap self-consistency).
3. **Sampling & decode** — generates frames autoregressively in
   latent space using K-step Euler denoising, then decodes back to
   pixels through the tokenizer decoder.

**Plots saved:**

| File | Description |
|------|-------------|
| `outputs/dynamics_ball_tok_loss.png` | Tokenizer pretraining loss (combined + eval MSE) vs step. |
| `outputs/dynamics_ball_loss.png` | Dynamics shortcut forcing loss (combined, flow MSE, bootstrap MSE) vs step. |
| `outputs/dynamics_ball_frames.png` | For each trajectory: ground-truth frames (TRUE), tokenizer reconstruction (RECON, grey border), and dynamics-generated frames (GEN). Green border = context frames fed as input; blue border = predicted frames from autoregressive rollout. |

*Tokenizer pretraining: combined RMS-normalized loss (blue) stabilises around 1.0; eval full-image MSE (orange) drops to ~2e-5, indicating near-perfect reconstruction.*

![dynamics_ball_tok_loss](outputs/dynamics_ball_tok_loss.png)

*Dynamics training: combined shortcut forcing loss (blue) drops from ~0.2 to ~0.006 over 3000 steps. Flow MSE (orange, dashed) and bootstrap MSE (green, dotted) both decrease steadily.*

![dynamics_ball_loss](outputs/dynamics_ball_loss.png)

*Four trajectories (8 frames each) with three rows per trajectory. TRUE: ground-truth frames. RECON (grey border): tokenizer-only encode-decode, confirming the latent space is faithful. GEN: dynamics model output decoded through the tokenizer — green-bordered frames are ground-truth context fed as input, blue-bordered frames are autoregressively generated predictions. The generated ball positions track the true trajectory.*

![dynamics_ball_frames](outputs/dynamics_ball_frames.png)

## Output directory

All plots are saved to `examples/outputs/`.  The directory is created
automatically on the first run.  These files are not tracked by version
control.
