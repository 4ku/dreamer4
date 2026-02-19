# Toy Examples for the Block-Causal Transformer

Four runnable scripts that verify the transformer works end-to-end on
synthetic tasks.  Each script trains a tiny model, asserts that the loss
drops significantly, and saves visualisation plots to the `outputs/`
directory.

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

## Output directory

All plots are saved to `examples/outputs/`.  The directory is created
automatically on the first run.  These files are not tracked by version
control.
