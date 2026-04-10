"""
Full Dreamer 4 Pipeline — all modules on a bouncing-ball task.

Exercises every implemented component end-to-end:
    Phase 1a: Tokenizer pretraining  (Module 2)
    Phase 1b: Dynamics pretraining   (Module 3 — shortcut forcing, n_agent=0)
    Phase 2:  Agent finetuning       (Module 4 — BC + reward prediction, n_agent=1)
    Phase 3:  Imagination training   (Module 4 — PMPO + TD-lambda value)
    Final:    Combined visualization

Run:
    cd dreamer4_pytorch
    python -m examples.full_pipeline
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dreamer4.dynamics import (
    DynamicsModel,
    shortcut_forcing_loss,
    sample_sequence,
    sample_flow_schedule,
    corrupt_representations,
    ramp_weight,
)
from dreamer4.tokenizer import (
    Decoder,
    Encoder,
    MAECompositedLPIPS,
    RMSLossNormalizer,
    Tokenizer,
    recon_loss_from_mae,
)
from dreamer4.utils import (
    patchify, unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)
from dreamer4.agent import (
    TaskEncoder, PolicyHead, RewardHead, ValueHead,
    behavior_cloning_loss, reward_prediction_loss,
    compute_lambda_returns,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

# --- Image / environment config ---
IMG_H, IMG_W, IMG_C = 32, 32, 3
PATCH_SIZE = 8
N_PATCHES = (IMG_H // PATCH_SIZE) * (IMG_W // PATCH_SIZE)   # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * IMG_C                  # 192
BALL_RADIUS = 3.0
BALL_COLOR = torch.tensor([1.0, 0.8, 0.2])

# --- Tokenizer config ---
D_MODEL_TOK = 256
N_LATENTS = 16
D_BOTTLENECK = 48

# --- Dynamics config ---
PACK_K = 1
N_SPATIAL = N_LATENTS // PACK_K       # 16
D_SPATIAL = D_BOTTLENECK * PACK_K     # 48
D_MODEL_DYN = 128
K_MAX = 4
ACTION_DIM = 2

# --- Agent config ---
MTP_LENGTH = 4
N_AGENT = 1


def _render_frames(
    B: int,
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate B trajectories of a bouncing ball, each with T frames.

    Returns:
        frames:  (B, T, C, H, W) float tensor in [0, 1].
        actions: (B, T, 2) float tensor (velocities).
        rewards: (B, T) float tensor (+1 if ball x > midpoint, else 0).
    """
    frames = torch.zeros(B, T, IMG_C, IMG_H, IMG_W, device=device)
    velocities = torch.zeros(B, T, 2, device=device)
    rewards = torch.zeros(B, T, device=device)

    yy = torch.arange(IMG_H, device=device, dtype=torch.float32)
    xx = torch.arange(IMG_W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    margin = BALL_RADIUS + 1
    px = torch.rand(B, device=device) * (IMG_W - 2 * margin) + margin
    py = torch.rand(B, device=device) * (IMG_H - 2 * margin) + margin

    speed = torch.rand(B, 2, device=device) * 1.5 + 1.5
    sign = torch.randint(0, 2, (B, 2), device=device).float() * 2 - 1
    vel = speed * sign

    mid_x = IMG_W / 2.0

    for t in range(T):
        velocities[:, t] = vel
        rewards[:, t] = (px > mid_x).float()

        dx = grid_x.unsqueeze(0) - px.view(B, 1, 1)
        dy = grid_y.unsqueeze(0) - py.view(B, 1, 1)
        dist_sq = dx * dx + dy * dy
        alpha = torch.clamp(1.0 - (dist_sq.sqrt() - BALL_RADIUS + 1.5) / 1.5, 0, 1)

        for c in range(IMG_C):
            frames[:, t, c] = alpha * BALL_COLOR[c]

        px = px + vel[:, 0]
        py = py + vel[:, 1]

        for coord, limit in [(px, IMG_W), (py, IMG_H)]:
            below = coord < BALL_RADIUS
            above = coord > limit - BALL_RADIUS - 1
            if below.any() or above.any():
                idx = 0 if coord is px else 1
                vel[:, idx] = torch.where(below | above, -vel[:, idx], vel[:, idx])
                coord.clamp_(BALL_RADIUS, limit - BALL_RADIUS - 1)

    return frames, velocities, rewards


# ======================================================================
# Phase 1a: Tokenizer pretraining
# ======================================================================

def phase_1a_tokenizer(device, train_patches, eval_patches):
    """Train the causal tokenizer with MSE + 0.2*LPIPS."""
    print("=" * 60)
    print("Phase 1a: Tokenizer pretraining (MSE + 0.2*LPIPS)")
    print("=" * 60)

    enc = Encoder(
        patch_dim=PATCH_DIM, d_model=D_MODEL_TOK, n_latents=N_LATENTS,
        n_patches=N_PATCHES, n_heads=4, depth=4, d_bottleneck=D_BOTTLENECK,
        time_every=1, mae_p_min=0.0, mae_p_max=0.9,
        use_qk_norm=False, logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=D_BOTTLENECK, d_model=D_MODEL_TOK, n_latents=N_LATENTS,
        n_patches=N_PATCHES, patch_dim=PATCH_DIM, n_heads=4, depth=4,
        time_every=1, use_qk_norm=False, logit_cap=None,
    )
    tok = Tokenizer(enc, dec).to(device)
    print(f"Tokenizer params: {sum(p.numel() for p in tok.parameters()):,}")

    lpips_fn = MAECompositedLPIPS(
        net="alex",
        H=IMG_H,
        W=IMG_W,
        C=IMG_C,
        patch_size=PATCH_SIZE,
        verbose=False,
    ).to(device).eval()
    loss_norm = RMSLossNormalizer(n_losses=2).to(device)
    optimizer = torch.optim.Adam(tok.parameters(), lr=3e-4)

    N_TRAIN = train_patches.shape[0]
    BATCH_SIZE = 32
    STEPS = 2000

    step_list, losses, eval_mses = [], [], []
    print(f"Training for {STEPS} steps...")

    for step in range(1, STEPS + 1):
        tok.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        batch = train_patches[idx]

        pred, mae_mask, _ = tok(batch)
        mse = recon_loss_from_mae(pred, batch, mae_mask)
        lp = lpips_fn(pred, batch, mae_mask)
        loss_norm.update(0, mse)
        loss_norm.update(1, lp)
        loss = loss_norm.normalize(0, mse) + 0.2 * loss_norm.normalize(1, lp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            step_list.append(step)
            losses.append(loss.item())
            tok.eval()
            with torch.no_grad():
                ev_pred, _, _ = tok(eval_patches)
                ev_mse = (ev_pred - eval_patches).pow(2).mean().item()
            eval_mses.append(ev_mse)

        if step % 500 == 0 or step == 1:
            print(f"  step {step:4d}  loss={loss.item():.5f}  eval_mse={eval_mses[-1]:.6f}")

    print(f"Final eval MSE: {eval_mses[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, losses, linewidth=1.5, label="Train (combined)", alpha=0.8)
    ax.semilogy(step_list, eval_mses, linewidth=1.5, label="Eval MSE", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 1a — Tokenizer Pretraining")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_pipeline_tok_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'full_pipeline_tok_loss.png'}")

    return tok


# ======================================================================
# Phase 1b: Dynamics pretraining (no agent tokens)
# ======================================================================

def phase_1b_dynamics(device, train_z, train_actions):
    """Train dynamics model with shortcut forcing (n_agent=0)."""
    print()
    print("=" * 60)
    print("Phase 1b: Dynamics pretraining (shortcut forcing)")
    print("=" * 60)

    dynamics = DynamicsModel(
        d_model=D_MODEL_DYN, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
        n_register=2, n_agent=0,
        n_heads=4, depth=6, k_max=K_MAX, action_dim=ACTION_DIM,
        time_every=2, dropout=0.0,
        use_qk_norm=False, logit_cap=None,
        space_mode="wm_agent_isolated",
    ).to(device)
    print(f"Dynamics (pretrain) params: {sum(p.numel() for p in dynamics.parameters()):,}")

    N_TRAIN = train_z.shape[0]
    BATCH_SIZE = 32
    STEPS = 4000

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    step_list, losses_all, losses_flow, losses_boot = [], [], [], []

    print(f"Training for {STEPS} steps...")
    for step in range(1, STEPS + 1):
        dynamics.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        loss, aux = shortcut_forcing_loss(
            dynamics, z1=train_z[idx], actions=train_actions[idx],
            k_max=K_MAX, bootstrap_fraction=0.25,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            step_list.append(step)
            losses_all.append(loss.item())
            losses_flow.append(aux["flow_mse"].item())
            losses_boot.append(aux["boot_mse"].item())

        if step % 1000 == 0 or step == 1:
            print(f"  step {step:4d}  loss={loss.item():.6f}  "
                  f"flow={aux['flow_mse'].item():.6f}  boot={aux['boot_mse'].item():.6f}")

    print(f"Final loss: {losses_all[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, losses_all, linewidth=1.5, label="Combined", alpha=0.9)
    ax.semilogy(step_list, losses_flow, linewidth=1.2, label="Flow", alpha=0.7, ls="--")
    ax.semilogy(step_list, losses_boot, linewidth=1.2, label="Bootstrap", alpha=0.7, ls=":")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 1b — Dynamics Pretraining")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_pipeline_dyn_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'full_pipeline_dyn_loss.png'}")

    return dynamics


# ======================================================================
# Phase 2: Agent finetuning (BC + reward prediction)
# ======================================================================

def phase_2_agent(device, pretrained_dyn, train_z, train_actions, train_rewards):
    """Finetune dynamics with agent tokens; train policy + reward heads."""
    print()
    print("=" * 60)
    print("Phase 2: Agent finetuning (BC + reward prediction)")
    print("=" * 60)

    dynamics = DynamicsModel(
        d_model=D_MODEL_DYN, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
        n_register=2, n_agent=N_AGENT,
        n_heads=4, depth=6, k_max=K_MAX, action_dim=ACTION_DIM,
        time_every=2, dropout=0.0,
        use_qk_norm=False, logit_cap=None,
        space_mode="wm_agent",
    ).to(device)

    # Copy pretrained weights (keys that exist in both)
    pretrained_sd = pretrained_dyn.state_dict()
    agent_sd = dynamics.state_dict()
    transferred = 0
    for k in agent_sd:
        if k in pretrained_sd and pretrained_sd[k].shape == agent_sd[k].shape:
            agent_sd[k] = pretrained_sd[k]
            transferred += 1
    dynamics.load_state_dict(agent_sd)
    print(f"Transferred {transferred}/{len(agent_sd)} parameters from pretrained dynamics")

    task_enc = TaskEncoder(num_tasks=1, d_model=D_MODEL_DYN, n_agent=N_AGENT).to(device)
    policy = PolicyHead(
        d_model=D_MODEL_DYN, action_dim=ACTION_DIM, action_type="continuous",
        mtp_length=MTP_LENGTH, mlp_depth=2, mlp_ratio=2.0, min_std=0.1,
    ).to(device)
    reward_head = RewardHead(
        d_model=D_MODEL_DYN, mtp_length=MTP_LENGTH,
        mlp_depth=2, mlp_ratio=2.0, num_bins=127,
        bin_low=-2.0, bin_high=2.0,
    ).to(device)

    all_params = (
        list(dynamics.parameters())
        + list(task_enc.parameters())
        + list(policy.parameters())
        + list(reward_head.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=2e-4)

    N_TRAIN = train_z.shape[0]
    BATCH_SIZE = 32
    STEPS = 2500

    step_list = []
    dyn_losses, bc_losses, rw_losses = [], [], []
    task_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    print(f"Training for {STEPS} steps (dynamics + BC + reward)...")
    for step in range(1, STEPS + 1):
        dynamics.train(); policy.train(); reward_head.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        z1_batch = train_z[idx]
        act_batch = train_actions[idx]
        rew_batch = train_rewards[idx]
        B, T = z1_batch.shape[:2]

        agent_tok = task_enc(task_ids[:B])          # (B, 1, N_AGENT, D)
        agent_tok = agent_tok.expand(B, T, -1, -1)  # (B, T, N_AGENT, D)

        # Dynamics forward with flow schedule (to get agent_out for heads)
        d_f, step_f, tau_f, sig_f = sample_flow_schedule(B, T, K_MAX, device)
        z_tilde, _ = corrupt_representations(z1_batch, tau_f)

        z1_hat, agent_out = dynamics(
            act_batch, step_f, sig_f, z_tilde,
            agent_tokens=agent_tok,
        )

        w = ramp_weight(tau_f)
        flow_per = (z1_hat - z1_batch).pow(2).mean(dim=(2, 3))
        dyn_loss = (flow_per * w).mean()

        # Agent heads use agent_out averaged over n_agent dim
        agent_embed = agent_out.mean(dim=2)  # (B, T, D_MODEL_DYN)

        bc_loss = behavior_cloning_loss(
            policy, agent_embed, act_batch, mtp_length=MTP_LENGTH,
        )
        rw_loss = reward_prediction_loss(
            reward_head, agent_embed, rew_batch, mtp_length=MTP_LENGTH,
        )

        loss = dyn_loss + bc_loss + rw_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            step_list.append(step)
            dyn_losses.append(dyn_loss.item())
            bc_losses.append(bc_loss.item())
            rw_losses.append(rw_loss.item())

        if step % 500 == 0 or step == 1:
            print(f"  step {step:4d}  dyn={dyn_loss.item():.5f}  "
                  f"bc={bc_loss.item():.4f}  reward={rw_loss.item():.4f}")

    print(f"Final: dyn={dyn_losses[-1]:.5f}  bc={bc_losses[-1]:.4f}  reward={rw_losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, dyn_losses, linewidth=1.5, label="Dynamics", alpha=0.9)
    ax.semilogy(step_list, bc_losses, linewidth=1.5, label="BC (policy)", alpha=0.8)
    ax.semilogy(step_list, rw_losses, linewidth=1.5, label="Reward pred", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 2 — Agent Finetuning")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_pipeline_agent_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'full_pipeline_agent_loss.png'}")

    return dynamics, task_enc, policy, reward_head


# ======================================================================
# Phase 3: Imagination training (PMPO + value)
# ======================================================================

def phase_3_imagination(
    device, dynamics, task_enc, policy, reward_head,
    train_z, train_actions, train_rewards,
):
    """Train value head and finetune policy via PMPO on imagined rollouts.

    Uses direct forward passes through the dynamics model on dataset
    contexts (with flow-schedule corruption) so the agent embeddings match
    the distribution the reward head was trained on.  This avoids the
    train/inference distribution mismatch that occurs when using
    sample_sequence with clean context.
    """
    print()
    print("=" * 60)
    print("Phase 3: Imagination training (PMPO + TD-lambda)")
    print("=" * 60)

    value_head = ValueHead(
        d_model=D_MODEL_DYN, mlp_depth=2, mlp_ratio=2.0, num_bins=127,
        bin_low=-10.0, bin_high=10.0,
    ).to(device)

    trainable = list(policy.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=3e-4)

    N_TRAIN = train_z.shape[0]
    T = train_z.shape[1]
    BATCH_SIZE = 32
    STEPS = 800

    dynamics.eval()
    task_enc.eval()
    reward_head.eval()

    step_list, pol_losses, val_losses = [], [], []
    task_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    print(f"Training for {STEPS} imagination steps...")
    for step in range(1, STEPS + 1):
        policy.train(); value_head.train()
        idx = torch.randint(0, N_TRAIN, (BATCH_SIZE,), device=device)
        z1_batch = train_z[idx]
        act_batch = train_actions[idx]
        rew_batch = train_rewards[idx]
        B = z1_batch.shape[0]

        agent_tok = task_enc(task_ids[:B])
        agent_tok = agent_tok.expand(B, T, -1, -1)

        # Run dynamics on dataset with flow-schedule corruption (same
        # distribution as Phase 2) to get agent embeddings.
        with torch.no_grad():
            d_f, step_f, tau_f, sig_f = sample_flow_schedule(B, T, K_MAX, device)
            z_tilde, _ = corrupt_representations(z1_batch, tau_f)
            _, agent_out = dynamics(
                act_batch, step_f, sig_f, z_tilde,
                agent_tokens=agent_tok,
            )

        agent_embed = agent_out.mean(dim=2)  # (B, T, D)

        # Use GT rewards for lambda-return targets.  In the full system
        # the reward head would be used, but in this toy setting we use
        # GT to keep the value signal clean and avoid cascading errors
        # from a reward head that hasn't perfectly converged.
        pred_rewards = rew_batch                              # (B, T)
        pred_values = value_head.predict(agent_embed)         # (B, T)
        continues = torch.ones_like(pred_rewards)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                pred_rewards, pred_values.detach(), continues,
                gamma=0.95, lam=0.90,
            )
            lambda_returns = lambda_returns.clamp(-10.0, 10.0)

        # Value loss: cross-entropy against lambda-return targets
        value_logits = value_head(agent_embed)  # (B, T, num_bins)
        val_loss = value_head.twohot.loss(
            value_logits.reshape(-1, value_logits.shape[-1]),
            lambda_returns.reshape(-1),
        )

        # Policy loss -- simplified reinforce with sign(advantage).
        # Equivalent to PMPO alpha=0.5 but bounded because we clamp
        # log_probs, preventing the unbounded divergence to -inf that
        # the raw PMPO objective suffers from.
        policy_params = policy.forward_all(agent_embed)
        lp = policy.log_prob(policy_params[0], act_batch)  # (B, T)
        lp = lp.clamp(min=-10.0)
        advantages = lambda_returns - pred_values.detach()

        pol_loss = -(lp * advantages.sign()).mean()

        loss = pol_loss + val_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % 5 == 0 or step == 1:
            step_list.append(step)
            pol_losses.append(pol_loss.item())
            val_losses.append(val_loss.item())

        if step % 100 == 0 or step == 1:
            print(f"  step {step:3d}  policy={pol_loss.item():.4f}  "
                  f"value={val_loss.item():.4f}  "
                  f"pred_rw=[{pred_rewards.min().item():.2f}, {pred_rewards.max().item():.2f}]  "
                  f"pred_v=[{pred_values.min().item():.2f}, {pred_values.max().item():.2f}]")

    print(f"Final: policy={pol_losses[-1]:.4f}  value={val_losses[-1]:.4f}")

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    color_pol = "#1565C0"
    ax1.plot(step_list, pol_losses, linewidth=1.5, label="Policy (PMPO)",
             alpha=0.9, color=color_pol)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Policy loss", color=color_pol, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=color_pol)

    ax2 = ax1.twinx()
    color_val = "#E65100"
    ax2.plot(step_list, val_losses, linewidth=1.5, label="Value (TD-lambda)",
             alpha=0.9, color=color_val)
    ax2.set_ylabel("Value loss", color=color_val, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=color_val)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("Phase 3 — Imagination Training")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_pipeline_imagination_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'full_pipeline_imagination_loss.png'}")

    return value_head


# ======================================================================
# Final visualization
# ======================================================================

def visualize(
    device, tok, dyn_pretrained, dyn_agent, task_enc,
    policy, reward_head, value_head,
    eval_frames, eval_z, eval_z_bot, eval_actions, eval_rewards,
):
    """Generate combined visualization of all pipeline outputs."""
    print()
    print("=" * 60)
    print("Final visualization")
    print("=" * 60)

    tok.eval(); dyn_pretrained.eval(); dyn_agent.eval()
    policy.eval(); reward_head.eval(); value_head.eval()

    N_CTX = 2
    T = eval_z.shape[1]
    HORIZON = T - N_CTX
    N_SHOW = min(4, eval_z.shape[0])

    task_ids = torch.zeros(N_SHOW, dtype=torch.long, device=device)
    agent_tok = task_enc(task_ids)
    agent_tok = agent_tok.expand(N_SHOW, T, -1, -1)

    with torch.no_grad():
        # Dynamics-only generation (pretrained, n_agent=0)
        gen_z_pre, _ = sample_sequence(
            dyn_pretrained,
            context=eval_z[:N_SHOW, :N_CTX],
            horizon=HORIZON,
            k_max=K_MAX, K=K_MAX,
            actions=eval_actions[:N_SHOW],
            tau_ctx=0.05,
        )
        gen_bot_pre = unpack_spatial_to_bottleneck(gen_z_pre, k=PACK_K).clamp(-1, 1)
        gen_img_pre = unpatchify(
            tok.decoder(gen_bot_pre), IMG_H, IMG_W, IMG_C, PATCH_SIZE,
        ).clamp(0, 1)

        # Agent-conditioned generation
        gen_z_agent, agent_outs = sample_sequence(
            dyn_agent,
            context=eval_z[:N_SHOW, :N_CTX],
            horizon=HORIZON,
            k_max=K_MAX, K=K_MAX,
            actions=eval_actions[:N_SHOW],
            tau_ctx=0.05,
            agent_tokens=agent_tok,
        )
        gen_bot_agent = unpack_spatial_to_bottleneck(gen_z_agent, k=PACK_K).clamp(-1, 1)
        gen_img_agent = unpatchify(
            tok.decoder(gen_bot_agent), IMG_H, IMG_W, IMG_C, PATCH_SIZE,
        ).clamp(0, 1)

        # Reward and value predictions: use dynamics.forward() with a
        # deterministic high-signal-level corruption (tau=0.95) so the
        # agent embeddings are consistent across runs and close to what
        # Phase 2 training saw, avoiding the noise of random schedules.
        eval_z_show = eval_z[:N_SHOW]          # (N_SHOW, T, N_sp, D_sp)
        eval_act_show = eval_actions[:N_SHOW]  # (N_SHOW, T, A)
        emax = int(round(math.log2(K_MAX)))
        tau_fixed = torch.full((N_SHOW, T), 0.95, device=device)
        sig_fixed = torch.full((N_SHOW, T), int(0.95 * K_MAX),
                               device=device, dtype=torch.long)
        step_fixed = torch.full((N_SHOW, T), emax,
                                device=device, dtype=torch.long)
        z_tilde, _ = corrupt_representations(eval_z_show, tau_fixed)
        _, agent_out_viz = dyn_agent(
            eval_act_show, step_fixed, sig_fixed, z_tilde,
            agent_tokens=agent_tok,
        )
        a_embed_viz = agent_out_viz.mean(dim=2)  # (N_SHOW, T, D)

        pred_rw_full = reward_head.predict(a_embed_viz)  # (N_SHOW, T)
        pred_val_full = value_head.predict(a_embed_viz)  # (N_SHOW, T)

        pred_rw = pred_rw_full[:, N_CTX:]   # (N_SHOW, HORIZON)
        pred_val = pred_val_full[:, N_CTX:]  # (N_SHOW, HORIZON)

        # Compute GT lambda-returns for the value plot reference.
        # Uses GT rewards + value bootstrap (same gamma/lam as Phase 3).
        gt_rw_full = eval_rewards[:N_SHOW]              # (N_SHOW, T)
        continues_full = torch.ones_like(gt_rw_full)
        gt_val_full = value_head.predict(a_embed_viz)   # use same embed
        gt_lambda_returns = compute_lambda_returns(
            gt_rw_full, gt_val_full, continues_full,
            gamma=0.95, lam=0.90,
        ).clamp(-10.0, 10.0)
        gt_lambda = gt_lambda_returns[:, N_CTX:].cpu()  # (N_SHOW, HORIZON)

    # Clamp predictions to sensible display ranges based on GT
    gt_rw_all = eval_rewards[:N_SHOW, N_CTX:]
    rw_lo = gt_rw_all.min().item() - 0.5
    rw_hi = gt_rw_all.max().item() + 0.5
    pred_rw = pred_rw.clamp(rw_lo, rw_hi).cpu()
    pred_val = pred_val.cpu()

    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec
    import matplotlib.lines as mlines

    n_cols = T
    n_traj = N_SHOW
    x_pos = list(range(N_CTX, T))

    # Each trajectory block has 5 rows: GT, DYN, AGENT, Reward, Value.
    # Image rows get height ratio 1; chart rows get height ratio 1.5.
    row_ratios_per_traj = [1, 1, 1, 1.5, 1.5]
    # Add a thin spacer row (0.3) between trajectory blocks.
    all_ratios: list[float] = []
    for b in range(n_traj):
        all_ratios.extend(row_ratios_per_traj)
        if b < n_traj - 1:
            all_ratios.append(0.3)

    total_rows = len(all_ratios)
    fig_w = 1.8 * n_cols + 2.5
    fig_h = sum(all_ratios) * 1.1 + 2.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        total_rows, n_cols, figure=fig,
        height_ratios=all_ratios,
        hspace=0.35, wspace=0.12,
        left=0.10, right=0.97, top=0.94, bottom=0.06,
    )

    def _row_idx(traj: int, local_row: int) -> int:
        """Map (trajectory, local row) to global gridspec row."""
        rows_per_block = len(row_ratios_per_traj)
        spacer_offset = traj  # one spacer between each block
        return traj * rows_per_block + spacer_offset + local_row

    stored_axes: dict[str, plt.Axes] = {}

    for b in range(n_traj):
        for t in range(n_cols):
            gt = eval_frames[b, t].permute(1, 2, 0).cpu().numpy()
            pr_pre = gen_img_pre[b, t].permute(1, 2, 0).cpu().numpy()
            pr_ag = gen_img_agent[b, t].permute(1, 2, 0).cpu().numpy()

            # Row 0: Ground truth
            ax_gt = fig.add_subplot(gs[_row_idx(b, 0), t])
            ax_gt.imshow(gt)
            ax_gt.set_xticks([]); ax_gt.set_yticks([])
            if b == 0:
                lbl = f"t={t} (ctx)" if t < N_CTX else f"t={t}"
                ax_gt.set_title(lbl, fontsize=8)
            if t == 0:
                stored_axes[f"gt_{b}"] = ax_gt

            # Row 1: Dynamics-only
            ax_pre = fig.add_subplot(gs[_row_idx(b, 1), t])
            ax_pre.imshow(pr_pre)
            ax_pre.set_xticks([]); ax_pre.set_yticks([])
            bc = "#4CAF50" if t < N_CTX else "#FF9800"
            for sp in ax_pre.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2)
            if t == 0:
                stored_axes[f"dyn_{b}"] = ax_pre

            # Row 2: Agent-conditioned
            ax_ag = fig.add_subplot(gs[_row_idx(b, 2), t])
            ax_ag.imshow(pr_ag)
            ax_ag.set_xticks([]); ax_ag.set_yticks([])
            bc = "#4CAF50" if t < N_CTX else "#1565C0"
            for sp in ax_ag.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2)
            if t == 0:
                stored_axes[f"ag_{b}"] = ax_ag

        # Row 3: Reward bar chart (spans all columns)
        ax_rw = fig.add_subplot(gs[_row_idx(b, 3), :])
        gt_rw = eval_rewards[b, N_CTX:].cpu().numpy()
        pr_rw = pred_rw[b].numpy()
        bar_w = 0.35
        ax_rw.bar([x - bar_w / 2 for x in x_pos], gt_rw, bar_w,
                   label="GT reward", color="#4CAF50", alpha=0.75)
        ax_rw.bar([x + bar_w / 2 for x in x_pos], pr_rw, bar_w,
                   label="Predicted reward", color="#1565C0", alpha=0.75)
        ax_rw.set_ylabel("Reward", fontsize=7)
        ax_rw.set_ylim(rw_lo - 0.1, rw_hi + 0.1)
        ax_rw.set_xticks(x_pos)
        ax_rw.set_xticklabels([f"t={t}" for t in x_pos], fontsize=6)
        ax_rw.tick_params(axis="y", labelsize=6)
        ax_rw.grid(axis="y", alpha=0.2)
        if b == 0:
            ax_rw.legend(fontsize=6, loc="upper right")
        stored_axes[f"rw_{b}"] = ax_rw

        # Row 4: Value prediction + GT lambda-return reference
        ax_val = fig.add_subplot(gs[_row_idx(b, 4), :])
        pv = pred_val[b].numpy()
        gl = gt_lambda[b].numpy()
        ax_val.plot(x_pos, gl, "s--", color="#4CAF50", markersize=4,
                    linewidth=1.2, alpha=0.8, label="GT lambda-return")
        ax_val.plot(x_pos, pv, "o-", color="#7B1FA2", markersize=4,
                    linewidth=1.5, label="Predicted value")
        ax_val.set_ylabel("Value", fontsize=7)
        ax_val.set_xticks(x_pos)
        ax_val.set_xticklabels([f"t={t}" for t in x_pos], fontsize=6)
        ax_val.tick_params(axis="y", labelsize=6)
        ax_val.grid(axis="y", alpha=0.2)
        if b == 0:
            ax_val.legend(fontsize=6, loc="upper right")
        stored_axes[f"val_{b}"] = ax_val

    # --- Row labels on the left side ---
    row_label_cfg = [
        ("gt_{b}",  "#{b} GT",    "black"),
        ("dyn_{b}", "#{b} Dyn",   "#FF9800"),
        ("ag_{b}",  "#{b} Agent", "#1565C0"),
        ("rw_{b}",  "#{b} R",     "#388E3C"),
        ("val_{b}", "#{b} V",     "#7B1FA2"),
    ]
    for b in range(n_traj):
        for key_tmpl, label_tmpl, color in row_label_cfg:
            key = key_tmpl.format(b=b)
            ax = stored_axes[key]
            pos = ax.get_position()
            fig.text(
                0.01, (pos.y0 + pos.y1) / 2,
                label_tmpl.format(b=b + 1),
                fontsize=7, fontweight="bold", color=color,
                va="center", ha="left",
            )

    fig.suptitle(
        "Full Dreamer 4 Pipeline — Bouncing Ball",
        fontsize=13, fontweight="bold",
    )

    # Bottom legend explaining border colors
    legend_handles = [
        Patch(facecolor="none", edgecolor="#4CAF50", linewidth=2,
              label="Context (GT input)"),
        Patch(facecolor="none", edgecolor="#FF9800", linewidth=2,
              label="Dynamics-only predicted"),
        Patch(facecolor="none", edgecolor="#1565C0", linewidth=2,
              label="Agent-conditioned predicted"),
        mlines.Line2D([], [], color="#4CAF50", marker="s", ls="--",
                       markersize=4, linewidth=1.2, label="GT lambda-return"),
        mlines.Line2D([], [], color="#7B1FA2", marker="o", ls="-",
                       markersize=4, linewidth=1.5, label="Predicted value"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=7, frameon=True, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.53, 0.001))

    fig.savefig(OUTPUT_DIR / "full_pipeline_frames.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_DIR / 'full_pipeline_frames.png'}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_TRAIN = 512
    N_EVAL = 32
    T = 8

    print("Generating data...")
    train_frames, train_vels, train_rewards = _render_frames(N_TRAIN, T, device)
    eval_frames, eval_vels, eval_rewards = _render_frames(N_EVAL, T, device)

    train_patches = patchify(train_frames, PATCH_SIZE)
    eval_patches = patchify(eval_frames, PATCH_SIZE)

    # Normalize actions
    vel_scale = train_vels.abs().max().item()
    if vel_scale > 0:
        train_actions = train_vels / vel_scale
        eval_actions = eval_vels / vel_scale
    else:
        train_actions = train_vels
        eval_actions = eval_vels

    # ---- Phase 1a: Tokenizer ----
    tok = phase_1a_tokenizer(device, train_patches, eval_patches)

    # ---- Encode data to latent space ----
    tok.eval()
    with torch.no_grad():
        train_z_bot = tok.encode(train_patches)
        eval_z_bot = tok.encode(eval_patches)

    train_z = pack_bottleneck_to_spatial(train_z_bot, n_spatial=N_SPATIAL, k=PACK_K)
    eval_z = pack_bottleneck_to_spatial(eval_z_bot, n_spatial=N_SPATIAL, k=PACK_K)
    print(f"Latent shape: {train_z.shape}  range [{train_z.min():.2f}, {train_z.max():.2f}]")

    # ---- Phase 1b: Dynamics pretraining ----
    dyn_pretrained = phase_1b_dynamics(device, train_z, train_actions)

    # ---- Phase 2: Agent finetuning ----
    dyn_agent, task_enc, policy, reward_head = phase_2_agent(
        device, dyn_pretrained, train_z, train_actions, train_rewards,
    )

    # ---- Phase 3: Imagination ----
    value_head = phase_3_imagination(
        device, dyn_agent, task_enc, policy, reward_head,
        train_z, train_actions, train_rewards,
    )

    # ---- Final visualization ----
    visualize(
        device, tok, dyn_pretrained, dyn_agent, task_enc,
        policy, reward_head, value_head,
        eval_frames, eval_z, eval_z_bot, eval_actions, eval_rewards,
    )

    print()
    print("=" * 60)
    print("All phases complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
