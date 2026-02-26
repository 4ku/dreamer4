"""
Dreamer 4 — full pipeline on DMControl (cartpole-swingup).

Exercises every implemented module end-to-end on a real visual RL task:

    Phase 0:  Data collection (random policy)
    Phase 1a: Tokenizer pretraining  (MSE + LPIPS)
    Phase 1b: Dynamics pretraining   (shortcut forcing)
    Phase 2:  Agent finetuning       (BC + reward prediction)
    Phase 3:  Imagination training   (PMPO + TD-lambda)
    Eval:     Run learned policy in environment

Run:
    cd /mnt/dsk1/dreamer4
    MUJOCO_GL=egl python -m examples.dmcontrol_imagination

Target: ~10 minutes on a single GPU with tiny model sizes.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import lpips as lpips_lib

from dreamer4.envs import DMControlEnv
from dreamer4.replay import ReplayBuffer
from dreamer4.dynamics import (
    DynamicsModel,
    shortcut_forcing_loss,
    sample_sequence,
    sample_flow_schedule,
    corrupt_representations,
    ramp_weight,
)
from dreamer4.tokenizer import (
    Encoder, Decoder, Tokenizer,
    recon_loss_from_mae, lpips_on_mae_recon, RMSLossNormalizer,
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
from dreamer4.imagination import (
    imagine_rollout,
    imagination_training_step,
    make_prior_policy,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

# --- Environment config ---
DOMAIN, TASK = "cartpole", "swingup"
IMG_SIZE = 64
IMG_C = 3
PATCH_SIZE = 8
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * IMG_C  # 192
ACTION_REPEAT = 2

# --- Tokenizer config ---
D_MODEL_TOK = 128
N_LATENTS = 8
D_BOTTLENECK = 32

# --- Dynamics config ---
PACK_K = 1
N_SPATIAL = N_LATENTS // PACK_K  # 8
D_SPATIAL = D_BOTTLENECK * PACK_K  # 32
D_MODEL_DYN = 64
K_MAX = 4
MTP_LENGTH = 4
N_AGENT = 1

# --- Data collection ---
N_EPISODES = 40
SEQ_LEN = 16
BATCH_SIZE = 16


# ======================================================================
# Phase 0: Data collection
# ======================================================================

def phase_0_collect(device):
    """Collect random-policy episodes into replay buffer."""
    print("=" * 60)
    print("Phase 0: Collecting random episodes")
    print("=" * 60)

    env = DMControlEnv(DOMAIN, TASK, size=IMG_SIZE, action_repeat=ACTION_REPEAT)
    action_dim = env.action_dim
    buf = ReplayBuffer(capacity=200_000)

    total_steps = 0
    total_reward = 0.0
    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.sample_random_action()
            next_obs, reward, done, _, _ = env.step(action)
            buf.add(obs, action, reward, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
        total_reward += ep_reward

    avg_reward = total_reward / N_EPISODES
    print(f"  Collected {N_EPISODES} episodes, {total_steps} steps")
    print(f"  Avg episode reward (random): {avg_reward:.2f}")
    return buf, action_dim


# ======================================================================
# Phase 1a: Tokenizer pretraining
# ======================================================================

def phase_1a_tokenizer(device, buf):
    """Train the causal tokenizer with MSE + 0.2*LPIPS."""
    print()
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
    print(f"  Tokenizer params: {sum(p.numel() for p in tok.parameters()):,}")

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    loss_norm = RMSLossNormalizer(n_losses=2).to(device)
    optimizer = torch.optim.Adam(tok.parameters(), lr=3e-4)

    STEPS = 2000
    step_list, losses = [], []

    print(f"  Training for {STEPS} steps...")
    for step in range(1, STEPS + 1):
        tok.train()
        batch_data = buf.sample_sequence(BATCH_SIZE, seq_len=4, device=device)
        frames = batch_data["obs"]  # (B, T, C, H, W)
        patches = patchify(frames, PATCH_SIZE)  # (B, T, N_PATCHES, PATCH_DIM)

        pred, mae_mask, _ = tok(patches)
        mse = recon_loss_from_mae(pred, patches, mae_mask)
        lp = lpips_on_mae_recon(
            lpips_fn, pred, patches, mae_mask,
            H=IMG_SIZE, W=IMG_SIZE, C=IMG_C, patch_size=PATCH_SIZE,
        )
        loss_norm.update(0, mse)
        loss_norm.update(1, lp)
        loss = loss_norm.normalize(0, mse) + 0.2 * loss_norm.normalize(1, lp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            step_list.append(step)
            losses.append(loss.item())

        if step % 500 == 0 or step == 1:
            print(f"    step {step:4d}  loss={loss.item():.5f}")

    print(f"  Final loss: {losses[-1]:.5f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, losses, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 1a — Tokenizer Pretraining (DMControl)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dmc_tok_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {OUTPUT_DIR / 'dmc_tok_loss.png'}")

    return tok


# ======================================================================
# Phase 1b: Dynamics pretraining
# ======================================================================

def phase_1b_dynamics(device, tok, buf, action_dim):
    """Train dynamics model with shortcut forcing."""
    print()
    print("=" * 60)
    print("Phase 1b: Dynamics pretraining (shortcut forcing)")
    print("=" * 60)

    dynamics = DynamicsModel(
        d_model=D_MODEL_DYN, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
        n_register=2, n_agent=0,
        n_heads=2, depth=4, k_max=K_MAX, action_dim=action_dim,
        time_every=2, dropout=0.0,
        use_qk_norm=False, logit_cap=None,
        space_mode="wm_agent_isolated",
    ).to(device)
    print(f"  Dynamics params: {sum(p.numel() for p in dynamics.parameters()):,}")

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    STEPS = 3000
    step_list, losses_all = [], []

    tok.eval()
    print(f"  Training for {STEPS} steps...")
    for step in range(1, STEPS + 1):
        dynamics.train()
        batch_data = buf.sample_sequence(BATCH_SIZE, seq_len=SEQ_LEN, device=device)
        frames = batch_data["obs"]  # (B, T, C, H, W)
        actions = batch_data["actions"]  # (B, T, A)

        with torch.no_grad():
            patches = patchify(frames, PATCH_SIZE)
            bottleneck = tok.encode(patches)
            z1 = pack_bottleneck_to_spatial(bottleneck, n_spatial=N_SPATIAL, k=PACK_K)

        loss, aux = shortcut_forcing_loss(
            dynamics, z1=z1, actions=actions,
            k_max=K_MAX, bootstrap_fraction=0.25,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            step_list.append(step)
            losses_all.append(loss.item())

        if step % 500 == 0 or step == 1:
            print(f"    step {step:4d}  loss={loss.item():.6f}")

    print(f"  Final loss: {losses_all[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, losses_all, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 1b — Dynamics Pretraining (DMControl)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dmc_dyn_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {OUTPUT_DIR / 'dmc_dyn_loss.png'}")

    return dynamics


# ======================================================================
# Phase 2: Agent finetuning (BC + reward)
# ======================================================================

def phase_2_agent(device, pretrained_dyn, tok, buf, action_dim):
    """Finetune dynamics with agent tokens; train policy + reward heads."""
    print()
    print("=" * 60)
    print("Phase 2: Agent finetuning (BC + reward)")
    print("=" * 60)

    dynamics = DynamicsModel(
        d_model=D_MODEL_DYN, d_spatial=D_SPATIAL, n_spatial=N_SPATIAL,
        n_register=2, n_agent=N_AGENT,
        n_heads=2, depth=4, k_max=K_MAX, action_dim=action_dim,
        time_every=2, dropout=0.0,
        use_qk_norm=False, logit_cap=None,
        space_mode="wm_agent",
    ).to(device)

    pretrained_sd = pretrained_dyn.state_dict()
    agent_sd = dynamics.state_dict()
    transferred = 0
    for k in agent_sd:
        if k in pretrained_sd and pretrained_sd[k].shape == agent_sd[k].shape:
            agent_sd[k] = pretrained_sd[k]
            transferred += 1
    dynamics.load_state_dict(agent_sd)
    print(f"  Transferred {transferred}/{len(agent_sd)} params from pretrained dynamics")

    task_enc = TaskEncoder(num_tasks=1, d_model=D_MODEL_DYN, n_agent=N_AGENT).to(device)
    policy = PolicyHead(
        d_model=D_MODEL_DYN, action_dim=action_dim, action_type="continuous",
        mtp_length=MTP_LENGTH, mlp_depth=2, mlp_ratio=2.0, min_std=0.1,
    ).to(device)
    reward_head = RewardHead(
        d_model=D_MODEL_DYN, mtp_length=MTP_LENGTH,
        mlp_depth=2, mlp_ratio=2.0, num_bins=63,
        bin_low=-5.0, bin_high=5.0,
    ).to(device)

    all_params = (
        list(dynamics.parameters())
        + list(task_enc.parameters())
        + list(policy.parameters())
        + list(reward_head.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=2e-4)

    STEPS = 2000
    step_list, dyn_losses, bc_losses, rw_losses = [], [], [], []
    task_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    tok.eval()
    print(f"  Training for {STEPS} steps...")
    for step in range(1, STEPS + 1):
        dynamics.train(); policy.train(); reward_head.train()
        batch_data = buf.sample_sequence(BATCH_SIZE, seq_len=SEQ_LEN, device=device)
        frames = batch_data["obs"]
        actions = batch_data["actions"]
        rewards = batch_data["rewards"]

        with torch.no_grad():
            patches = patchify(frames, PATCH_SIZE)
            bottleneck = tok.encode(patches)
            z1 = pack_bottleneck_to_spatial(bottleneck, n_spatial=N_SPATIAL, k=PACK_K)

        B, T = z1.shape[:2]
        agent_tok = task_enc(task_ids[:B]).expand(B, T, -1, -1)

        d_f, step_f, tau_f, sig_f = sample_flow_schedule(B, T, K_MAX, device)
        z_tilde, _ = corrupt_representations(z1, tau_f)
        z1_hat, agent_out = dynamics(
            actions, step_f, sig_f, z_tilde,
            agent_tokens=agent_tok,
        )

        w = ramp_weight(tau_f)
        flow_per = (z1_hat - z1).pow(2).mean(dim=(2, 3))
        dyn_loss = (flow_per * w).mean()

        agent_embed = agent_out.mean(dim=2)
        bc_loss = behavior_cloning_loss(policy, agent_embed, actions, mtp_length=MTP_LENGTH)
        rw_loss = reward_prediction_loss(reward_head, agent_embed, rewards, mtp_length=MTP_LENGTH)

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
            print(f"    step {step:4d}  dyn={dyn_loss.item():.5f}  "
                  f"bc={bc_loss.item():.4f}  rw={rw_loss.item():.4f}")

    print(f"  Final: dyn={dyn_losses[-1]:.5f}  bc={bc_losses[-1]:.4f}  rw={rw_losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(step_list, dyn_losses, linewidth=1.5, label="Dynamics", alpha=0.9)
    ax.semilogy(step_list, bc_losses, linewidth=1.5, label="BC", alpha=0.8)
    ax.semilogy(step_list, rw_losses, linewidth=1.5, label="Reward", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title("Phase 2 — Agent Finetuning (DMControl)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dmc_agent_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {OUTPUT_DIR / 'dmc_agent_loss.png'}")

    return dynamics, task_enc, policy, reward_head


# ======================================================================
# Phase 3: Imagination training
# ======================================================================

def phase_3_imagination(
    device, dynamics, task_enc, policy, reward_head, tok, buf, action_dim,
):
    """Train value head and finetune policy via imagination rollouts."""
    print()
    print("=" * 60)
    print("Phase 3: Imagination training (PMPO + TD-lambda)")
    print("=" * 60)

    value_head = ValueHead(
        d_model=D_MODEL_DYN, mlp_depth=2, mlp_ratio=2.0, num_bins=63,
        bin_low=-10.0, bin_high=10.0,
    ).to(device)

    prior_policy = make_prior_policy(policy)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value_head.parameters(), lr=3e-4)

    STEPS = 500
    HORIZON = 10
    IMAGINE_BATCH = 8

    dynamics.eval()
    task_enc.eval()
    reward_head.eval()
    tok.eval()

    step_list, pol_losses, val_losses, mean_rw = [], [], [], []

    print(f"  Training for {STEPS} imagination steps...")
    for step in range(1, STEPS + 1):
        policy.train(); value_head.train()

        batch_data = buf.sample_sequence(IMAGINE_BATCH, seq_len=4, device=device)
        frames = batch_data["obs"]
        ctx_actions = batch_data["actions"]

        with torch.no_grad():
            patches = patchify(frames, PATCH_SIZE)
            bottleneck = tok.encode(patches)
            z_ctx = pack_bottleneck_to_spatial(bottleneck, n_spatial=N_SPATIAL, k=PACK_K)

        traj = imagine_rollout(
            dynamics, policy, reward_head, value_head, task_enc,
            context=z_ctx,
            context_actions=ctx_actions,
            horizon=HORIZON,
            k_max=K_MAX,
            K=K_MAX,
            tau_ctx=0.1,
        )

        result = imagination_training_step(
            traj, policy, value_head,
            policy_optim, value_optim,
            prior_policy=prior_policy,
            gamma=0.99,
            lam=0.95,
            alpha=0.5,
            beta=0.3,
            max_grad_norm=1.0,
        )

        if step % 5 == 0 or step == 1:
            step_list.append(step)
            pol_losses.append(result["policy_loss"])
            val_losses.append(result["value_loss"])
            mean_rw.append(result["mean_reward"])

        if step % 100 == 0 or step == 1:
            print(f"    step {step:3d}  pol={result['policy_loss']:.4f}  "
                  f"val={result['value_loss']:.4f}  "
                  f"rw={result['mean_reward']:.3f}  "
                  f"v={result['mean_value']:.3f}")

    print(f"  Final: pol={pol_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    c_pol = "#1565C0"
    ax1.plot(step_list, pol_losses, linewidth=1.5, label="Policy", alpha=0.9, color=c_pol)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Policy loss", color=c_pol, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=c_pol)

    ax2 = ax1.twinx()
    c_val = "#E65100"
    ax2.plot(step_list, val_losses, linewidth=1.5, label="Value", alpha=0.9, color=c_val)
    ax2.set_ylabel("Value loss", color=c_val, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=c_val)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("Phase 3 — Imagination Training (DMControl)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dmc_imagination_loss.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {OUTPUT_DIR / 'dmc_imagination_loss.png'}")

    return value_head


# ======================================================================
# Evaluation: run policy in environment
# ======================================================================

def evaluate_policy(
    device, tok, dynamics, task_enc, policy, action_dim,
    n_episodes: int = 5,
):
    """Run the learned policy in the real environment and report returns."""
    print()
    print("=" * 60)
    print("Evaluation: running policy in environment")
    print("=" * 60)

    env = DMControlEnv(DOMAIN, TASK, size=IMG_SIZE, action_repeat=ACTION_REPEAT)
    tok.eval(); dynamics.eval(); task_enc.eval(); policy.eval()

    episode_returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        obs_history: list[torch.Tensor] = [obs.to(device)]
        act_history: list[torch.Tensor] = [torch.zeros(action_dim, device=device)]

        task_ids = torch.zeros(1, dtype=torch.long, device=device)

        step = 0
        while not done and step < 500:
            with torch.no_grad():
                obs_stack = torch.stack(obs_history[-8:]).unsqueeze(0)  # (1, T, C, H, W)
                patches = patchify(obs_stack, PATCH_SIZE)
                bottleneck = tok.encode(patches)
                z = pack_bottleneck_to_spatial(bottleneck, n_spatial=N_SPATIAL, k=PACK_K)

                T_cur = z.shape[1]
                agent_tok = task_enc(task_ids).expand(1, T_cur, -1, -1)

                emax = int(round(math.log2(K_MAX)))
                tau_fixed = torch.full((1, T_cur), 0.95, device=device)
                sig_fixed = torch.full((1, T_cur), int(0.95 * K_MAX),
                                       device=device, dtype=torch.long)
                step_fixed = torch.full((1, T_cur), emax,
                                        device=device, dtype=torch.long)
                z_tilde, _ = corrupt_representations(z, tau_fixed)

                act_stack = torch.stack(act_history[-8:]).unsqueeze(0)  # (1, T, A)

                _, agent_out = dynamics(
                    act_stack, step_fixed, sig_fixed, z_tilde,
                    agent_tokens=agent_tok,
                )
                agent_embed = agent_out[:, -1].mean(dim=-2)  # (1, D)

                params = policy.forward(agent_embed, head_idx=0)
                action, _ = policy.sample(params)
                action = action.squeeze(0).clamp(-1.0, 1.0)

            obs, reward, done, _, _ = env.step(action)
            obs_history.append(obs.to(device))
            act_history.append(action)
            ep_return += reward
            step += 1

        episode_returns.append(ep_return)
        print(f"  Episode {ep+1}/{n_episodes}: return={ep_return:.2f} ({step} steps)")

    avg = sum(episode_returns) / len(episode_returns)
    print(f"  Average return: {avg:.2f}")
    return episode_returns


# ======================================================================
# Visualization
# ======================================================================

def visualize(
    device, tok, dyn_pretrained, dyn_agent, task_enc,
    policy, reward_head, value_head, buf, action_dim,
):
    """Generate visualization of model predictions."""
    print()
    print("=" * 60)
    print("Generating visualizations")
    print("=" * 60)

    tok.eval(); dyn_pretrained.eval(); dyn_agent.eval()
    policy.eval(); reward_head.eval(); value_head.eval()

    N_CTX = 2
    VIS_HORIZON = 8
    N_SHOW = 3

    batch_data = buf.sample_sequence(N_SHOW, seq_len=N_CTX + VIS_HORIZON, device=device)
    frames = batch_data["obs"]          # (N, T, C, H, W)
    actions = batch_data["actions"]     # (N, T, A)
    rewards = batch_data["rewards"]     # (N, T)

    with torch.no_grad():
        patches = patchify(frames, PATCH_SIZE)
        bottleneck = tok.encode(patches)
        z = pack_bottleneck_to_spatial(bottleneck, n_spatial=N_SPATIAL, k=PACK_K)

        gt_bot = unpack_spatial_to_bottleneck(z, k=PACK_K).clamp(-1, 1)
        gt_recon = unpatchify(
            tok.decoder(gt_bot), IMG_SIZE, IMG_SIZE, IMG_C, PATCH_SIZE,
        ).clamp(0, 1)

        gen_z, _ = sample_sequence(
            dyn_pretrained,
            context=z[:, :N_CTX],
            horizon=VIS_HORIZON,
            k_max=K_MAX, K=K_MAX,
            actions=actions,
            tau_ctx=0.05,
        )
        gen_bot = unpack_spatial_to_bottleneck(gen_z, k=PACK_K).clamp(-1, 1)
        gen_img = unpatchify(
            tok.decoder(gen_bot), IMG_SIZE, IMG_SIZE, IMG_C, PATCH_SIZE,
        ).clamp(0, 1)

    T_total = N_CTX + VIS_HORIZON
    fig, axes = plt.subplots(
        N_SHOW * 3, T_total, figsize=(T_total * 1.4, N_SHOW * 3 * 1.4),
    )
    if N_SHOW * 3 == 1:
        axes = axes.reshape(1, -1)

    for n in range(N_SHOW):
        for t in range(T_total):
            ax_gt = axes[n * 3 + 0, t]
            ax_recon = axes[n * 3 + 1, t]
            ax_gen = axes[n * 3 + 2, t]

            img_gt = frames[n, t].permute(1, 2, 0).cpu().clamp(0, 1)
            img_recon = gt_recon[n, t].permute(1, 2, 0).cpu().clamp(0, 1)
            img_gen = gen_img[n, t].permute(1, 2, 0).cpu().clamp(0, 1)

            ax_gt.imshow(img_gt)
            ax_recon.imshow(img_recon)
            ax_gen.imshow(img_gen)

            border_color = "green" if t < N_CTX else "blue"
            for ax in [ax_gt, ax_recon, ax_gen]:
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)

            if t == 0:
                ax_gt.set_ylabel(f"#{n+1} GT", fontsize=8)
                ax_recon.set_ylabel(f"#{n+1} Recon", fontsize=8)
                ax_gen.set_ylabel(f"#{n+1} Gen", fontsize=8)

    fig.suptitle("DMControl Cartpole-Swingup: GT / Reconstruction / Generation",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dmc_frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {OUTPUT_DIR / 'dmc_frames.png'}")


# ======================================================================
# Main
# ======================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    buf, action_dim = phase_0_collect(device)

    tok = phase_1a_tokenizer(device, buf)

    dyn_pretrained = phase_1b_dynamics(device, tok, buf, action_dim)

    dyn_agent, task_enc, policy, reward_head = phase_2_agent(
        device, dyn_pretrained, tok, buf, action_dim,
    )

    value_head = phase_3_imagination(
        device, dyn_agent, task_enc, policy, reward_head, tok, buf, action_dim,
    )

    episode_returns = evaluate_policy(
        device, tok, dyn_agent, task_enc, policy, action_dim,
        n_episodes=5,
    )

    visualize(
        device, tok, dyn_pretrained, dyn_agent, task_enc,
        policy, reward_head, value_head, buf, action_dim,
    )

    print()
    print("=" * 60)
    print("Done! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
