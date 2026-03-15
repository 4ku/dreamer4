#!/usr/bin/env python3
"""
Dreamer 4 -- Web-based interactive world model explorer.

Serves a web page that shows the actual environment side-by-side with the
world model's predicted next frame.  Communication is via WebSocket so
the experience is real-time even over SSH without X forwarding.

Usage:
    cd /mnt/dsk1/dreamer4
    source .venv/bin/activate
    MUJOCO_GL=egl python scripts/play_web.py \
        --checkpoint runs/cartpole_balance/checkpoints/final.pt

    # Then open http://<server-ip>:8765 in your browser.
"""
import argparse
import base64
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreamer4.checkpoint import load_checkpoint
from dreamer4.config import TrainConfig, load_config
from dreamer4.dynamics import sample_one_timestep
from dreamer4.envs import DMControlEnv
from dreamer4.tokenizer import Decoder, Encoder, Tokenizer
from dreamer4.train import Dreamer4Agent
from dreamer4.utils import unpack_spatial_to_bottleneck, unpatchify

logger = logging.getLogger(__name__)

DISPLAY_SIZE = 256


def make_tokenizer(cfg: TrainConfig) -> Tokenizer:
    img_size = cfg.image_size
    patch_size = cfg.patch_size
    n_patches = (img_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * cfg.channels

    tok_d_model = 128
    tok_depth = 4
    tok_heads = 4

    enc = Encoder(
        patch_dim=patch_dim, d_model=tok_d_model,
        n_latents=cfg.n_bottleneck, n_patches=n_patches,
        n_heads=tok_heads, depth=tok_depth,
        d_bottleneck=cfg.d_bottleneck, time_every=1,
        mae_p_min=0.0, mae_p_max=0.9,
        use_qk_norm=False, logit_cap=None,
    )
    dec = Decoder(
        d_bottleneck=cfg.d_bottleneck, d_model=tok_d_model,
        n_latents=cfg.n_bottleneck, n_patches=n_patches,
        patch_dim=patch_dim, n_heads=tok_heads,
        depth=tok_depth, time_every=1,
        use_qk_norm=False, logit_cap=None,
    )
    return Tokenizer(enc, dec)


def numpy_to_jpeg_b64(arr: np.ndarray, size: int = DISPLAY_SIZE) -> str:
    img = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def numpy_to_jpeg_bytes(arr: np.ndarray, size: int = DISPLAY_SIZE) -> bytes:
    img = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def decode_latent_to_image(
    agent: Dreamer4Agent,
    z: torch.Tensor,
    cfg: TrainConfig,
) -> np.ndarray:
    """Decode a single latent (n_spatial, d_spatial) to an RGB numpy image."""
    z_4d = z.unsqueeze(0).unsqueeze(1)  # (1, 1, Nz, Dz)
    bottleneck = unpack_spatial_to_bottleneck(z_4d, k=cfg.packing_k).clamp(-1, 1)
    patch_pred = agent.tokenizer.decoder(bottleneck)
    img_pred = unpatchify(
        patch_pred, cfg.image_size, cfg.image_size, cfg.channels, cfg.patch_size,
    ).clamp(0, 1)
    return (img_pred[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def dream_step(
    agent: Dreamer4Agent,
    dream_state: dict,
    action: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Advance the dream by one step given an action.

    dream_state holds a running list of latent frames (z_frames) and
    actions (z_actions) that grows each call.  Only the initial z_frame
    comes from a real observation; the rest are the world model's own
    predictions.
    """
    max_ctx = cfg.policy_max_context

    dream_state["z_actions"].append(action.unsqueeze(0).to(device))

    z_frames = dream_state["z_frames"][-max_ctx:]
    z_actions = dream_state["z_actions"][-max_ctx:]

    past = torch.stack(z_frames, dim=1)  # (1, T, Nz, Dz)
    T_cur = past.shape[1]

    task_ids = torch.zeros(1, dtype=torch.long, device=device)
    agent_tok = agent.task_encoder(task_ids).expand(1, T_cur + 1, -1, -1)

    act_t = torch.stack(z_actions, dim=1)  # (1, T_act, A)
    if act_t.shape[1] < T_cur + 1:
        pad = torch.zeros(
            1, T_cur + 1 - act_t.shape[1], agent.action_dim, device=device,
        )
        act_t = torch.cat([act_t, pad], dim=1)

    z_next, a_out = sample_one_timestep(
        agent.dynamics, past_packed=past, k_max=cfg.k_max,
        K=cfg.imagination_K, actions=act_t,
        tau_ctx=cfg.imagination_tau_ctx, agent_tokens=agent_tok,
    )

    dream_state["z_frames"].append(z_next)  # (B, Nz, Dz), same as initial

    pred_reward = 0.0
    pred_value = 0.0
    if a_out is not None:
        agent_embed = a_out.mean(dim=-2)
        pred_reward = agent.reward_head.predict(agent_embed).item()
        pred_value = agent.value_head.predict(agent_embed).item()

    img_np = decode_latent_to_image(agent, z_next[0], cfg)
    return img_np, pred_reward, pred_value


# ---------------------------------------------------------------------------
# HTML / JS frontend (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dreamer 4 — World Model Explorer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 24px 16px;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; color: #a0c4ff; }
  .panels {
    display: flex; gap: 12px; margin-bottom: 16px;
    flex-wrap: wrap; justify-content: center;
  }
  .panel {
    display: flex; flex-direction: column; align-items: center;
  }
  .panel-label {
    font-size: 13px; font-weight: 600; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 1px; color: #888;
  }
  .panel img {
    width: 280px; height: 280px; border-radius: 8px;
    border: 2px solid #333; image-rendering: pixelated;
    background: #111;
  }
  .metrics {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 16px;
    background: #16213e; border-radius: 8px; padding: 12px 20px;
    margin-bottom: 16px; min-width: 420px;
  }
  .metric { text-align: center; }
  .metric .label { font-size: 11px; color: #666; text-transform: uppercase; }
  .metric .value { font-size: 16px; font-weight: 700; color: #bdc3c7; }
  .metric .value.mode-keyboard { color: #f1c40f; }
  .metric .value.mode-policy { color: #2ecc71; }
  .controls {
    display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
    justify-content: center;
  }
  button {
    padding: 10px 20px; border: 1px solid #444; border-radius: 6px;
    background: #16213e; color: #e0e0e0; font-family: inherit;
    font-size: 14px; cursor: pointer; transition: all 0.15s;
    min-width: 64px;
  }
  button:hover { background: #1a3a5c; border-color: #666; }
  button:active, button.active { background: #2a5a8c; border-color: #a0c4ff; }
  button.key-left, button.key-right { font-size: 20px; padding: 10px 24px; }
  .help {
    font-size: 12px; color: #555; text-align: center; max-width: 520px;
    line-height: 1.6;
  }
  .status {
    font-size: 12px; margin-top: 12px; padding: 4px 12px;
    border-radius: 4px;
  }
  .status.connected { color: #2ecc71; }
  .status.disconnected { color: #e74c3c; }
</style>
</head>
<body>
  <h1>Dreamer 4 — World Model Explorer</h1>

  <div class="panels">
    <div class="panel">
      <div class="panel-label">Environment</div>
      <img id="env-img" alt="env">
    </div>
    <div class="panel">
      <div class="panel-label">World Model</div>
      <img id="wm-img" alt="wm">
    </div>
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">Mode</div><div class="value" id="m-mode">KEYBOARD</div></div>
    <div class="metric"><div class="label">Step</div><div class="value" id="m-step">0</div></div>
    <div class="metric"><div class="label">Return</div><div class="value" id="m-return">0.0</div></div>
    <div class="metric"><div class="label">Reward</div><div class="value" id="m-reward">-</div></div>
    <div class="metric"><div class="label">Pred R</div><div class="value" id="m-pred-r">-</div></div>
    <div class="metric"><div class="label">Pred V</div><div class="value" id="m-pred-v">-</div></div>
  </div>

  <div class="controls">
    <button class="key-left" id="btn-left">&larr;</button>
    <button id="btn-none">Coast</button>
    <button class="key-right" id="btn-right">&rarr;</button>
    <button id="btn-policy">Policy (P)</button>
    <button id="btn-reset">Reset (R)</button>
  </div>

  <div class="help">
    <b>Keyboard:</b> Left/Right arrows or A/D to push cart &middot;
    P to toggle learned policy &middot; R to reset &middot; Space to coast<br>
    <b>Touch:</b> Use the buttons above
  </div>

  <div class="status disconnected" id="status">Connecting...</div>

<script>
const envImg = document.getElementById('env-img');
const wmImg = document.getElementById('wm-img');
const statusEl = document.getElementById('status');
const mMode = document.getElementById('m-mode');
const mStep = document.getElementById('m-step');
const mReturn = document.getElementById('m-return');
const mReward = document.getElementById('m-reward');
const mPredR = document.getElementById('m-pred-r');
const mPredV = document.getElementById('m-pred-v');

let ws = null;
let currentAction = 'none';
let usePolicy = false;
let connected = false;
let tickId = null;
let mode = 'ws';  // 'ws' or 'http'
let httpBusy = false;

const keysDown = new Set();

let prevEnvBlob = null;
let prevWmBlob = null;

async function loadImg(el, url, prevBlob) {
  try {
    const r = await fetch(url);
    if (!r.ok) return prevBlob;
    const blob = await r.blob();
    const objUrl = URL.createObjectURL(blob);
    el.src = objUrl;
    if (prevBlob) URL.revokeObjectURL(prevBlob);
    return objUrl;
  } catch(e) { return prevBlob; }
}

function handlePayload(d) {
  if (d.frame_id != null) {
    loadImg(envImg, '/img/env.jpg?t=' + d.frame_id, prevEnvBlob).then(u => { prevEnvBlob = u; });
    loadImg(wmImg, '/img/wm.jpg?t=' + d.frame_id, prevWmBlob).then(u => { prevWmBlob = u; });
  }
  mStep.textContent = d.ep_steps;
  mReturn.textContent = d.ep_return.toFixed(1);
  mReward.textContent = d.reward.toFixed(3);
  mPredR.textContent = d.pred_reward.toFixed(3);
  mPredV.textContent = d.pred_value.toFixed(2);
  const m = d.mode;
  mMode.textContent = m;
  mMode.className = 'value mode-' + m.toLowerCase();
  if (d.done) {
    statusEl.textContent = `Episode done (return=${d.ep_return.toFixed(1)}) — Connected (${mode.toUpperCase()})`;
  }
}

function startTicking() {
  if (tickId) clearInterval(tickId);
  tickId = setInterval(tick, 100);
}

function connectWS() {
  try {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
  } catch (e) {
    console.warn('WebSocket constructor failed, falling back to HTTP', e);
    switchToHTTP();
    return;
  }

  const wsTimeout = setTimeout(() => {
    if (ws && ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket open timed out, falling back to HTTP');
      ws.close();
      switchToHTTP();
    }
  }, 3000);

  ws.onopen = () => {
    clearTimeout(wsTimeout);
    mode = 'ws';
    connected = true;
    statusEl.textContent = 'Connected (WS)';
    statusEl.className = 'status connected';
    startTicking();
  };

  ws.onmessage = (evt) => { handlePayload(JSON.parse(evt.data)); };

  ws.onclose = () => {
    clearTimeout(wsTimeout);
    connected = false;
    if (tickId) { clearInterval(tickId); tickId = null; }
    if (mode === 'ws') {
      statusEl.textContent = 'Disconnected — trying HTTP fallback...';
      statusEl.className = 'status disconnected';
      switchToHTTP();
    }
  };

  ws.onerror = () => {
    clearTimeout(wsTimeout);
    ws.close();
  };
}

function switchToHTTP() {
  mode = 'http';
  connected = true;
  statusEl.textContent = 'Connected (HTTP polling)';
  statusEl.className = 'status connected';
  startTicking();
}

async function httpStep(action) {
  if (httpBusy) return;
  httpBusy = true;
  try {
    const resp = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action }),
    });
    if (resp.ok) {
      handlePayload(await resp.json());
      if (!connected) {
        connected = true;
        statusEl.textContent = 'Connected (HTTP polling)';
        statusEl.className = 'status connected';
      }
    }
  } catch (e) {
    console.warn('HTTP step failed', e);
  } finally {
    httpBusy = false;
  }
}

function getAction() {
  if (usePolicy) return 'policy';
  if (keysDown.has('ArrowLeft') || keysDown.has('KeyA')) return 'left';
  if (keysDown.has('ArrowRight') || keysDown.has('KeyD')) return 'right';
  return currentAction;
}

function tick() {
  const action = getAction();
  if (mode === 'ws') {
    if (!connected || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action }));
  } else {
    httpStep(action);
  }
  if (currentAction !== 'none' && !keysDown.has('ArrowLeft') && !keysDown.has('ArrowRight')
      && !keysDown.has('KeyA') && !keysDown.has('KeyD')) {
    currentAction = 'none';
  }
}

function sendAction(action) {
  if (mode === 'ws' && connected && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action }));
  } else if (mode === 'http') {
    httpStep(action);
  }
}

document.addEventListener('keydown', (e) => {
  keysDown.add(e.code);
  if (e.code === 'KeyP') {
    usePolicy = !usePolicy;
    document.getElementById('btn-policy').classList.toggle('active', usePolicy);
  }
  if (e.code === 'KeyR') sendAction('reset');
});

document.addEventListener('keyup', (e) => { keysDown.delete(e.code); });

document.getElementById('btn-left').addEventListener('mousedown', () => { currentAction = 'left'; });
document.getElementById('btn-left').addEventListener('mouseup', () => { currentAction = 'none'; });
document.getElementById('btn-left').addEventListener('touchstart', (e) => { e.preventDefault(); currentAction = 'left'; });
document.getElementById('btn-left').addEventListener('touchend', () => { currentAction = 'none'; });

document.getElementById('btn-right').addEventListener('mousedown', () => { currentAction = 'right'; });
document.getElementById('btn-right').addEventListener('mouseup', () => { currentAction = 'none'; });
document.getElementById('btn-right').addEventListener('touchstart', (e) => { e.preventDefault(); currentAction = 'right'; });
document.getElementById('btn-right').addEventListener('touchend', () => { currentAction = 'none'; });

document.getElementById('btn-none').addEventListener('click', () => { currentAction = 'none'; });

document.getElementById('btn-policy').addEventListener('click', () => {
  usePolicy = !usePolicy;
  document.getElementById('btn-policy').classList.toggle('active', usePolicy);
});

document.getElementById('btn-reset').addEventListener('click', () => { sendAction('reset'); });

connectWS();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def build_app(
    agent: Dreamer4Agent | None,
    cfg: TrainConfig,
    device: torch.device,
    has_checkpoint: bool,
):
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    app = FastAPI(title="Dreamer 4 World Model Explorer")

    env = DMControlEnv(
        cfg.domain, cfg.task,
        size=cfg.image_size,
        action_repeat=cfg.action_repeat,
    )

    state = {
        "obs_history": [],
        "act_history": [],
        "ep_return": 0.0,
        "ep_steps": 0,
        "use_policy": False,
        "env_jpeg": b"",
        "wm_jpeg": b"",
        "frame_id": 0,
        "dream": None,  # {"z_frames": [...], "z_actions": [...]}
    }

    @torch.no_grad()
    def reset_env():
        obs = env.reset()
        state["obs_history"] = [obs]
        state["act_history"] = []
        state["ep_return"] = 0.0
        state["ep_steps"] = 0
        env_rgb = env.render_rgb(size=cfg.image_size)
        state["env_jpeg"] = numpy_to_jpeg_bytes(env_rgb)
        state["wm_jpeg"] = state["env_jpeg"]

        if agent is not None:
            obs_t = obs.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)
            z0 = agent.encode_obs(obs_t)  # (1, 1, Nz, Dz)
            state["dream"] = {
                "z_frames": [z0[:, 0]],  # list of (1, Nz, Dz)
                "z_actions": [],
            }

    reset_env()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @app.get("/img/env.jpg")
    async def get_env_img():
        return Response(content=state["env_jpeg"], media_type="image/jpeg")

    @app.get("/img/wm.jpg")
    async def get_wm_img():
        return Response(content=state["wm_jpeg"], media_type="image/jpeg")

    def process_action(action_str: str) -> dict:
        """Shared logic for both WebSocket and HTTP polling."""
        if action_str == "reset":
            reset_env()
            state["frame_id"] += 1
            return {
                "frame_id": state["frame_id"],
                "reward": 0.0,
                "pred_reward": 0.0,
                "pred_value": 0.0,
                "ep_return": 0.0,
                "ep_steps": 0,
                "mode": "KEYBOARD",
                "done": False,
            }

        if action_str == "policy":
            state["use_policy"] = True
        else:
            state["use_policy"] = False

        if state["use_policy"] and agent is not None and has_checkpoint:
            action = agent.policy_action(
                state["obs_history"], state["act_history"],
            )
        else:
            act_val = 0.0
            if action_str == "left":
                act_val = -1.0
            elif action_str == "right":
                act_val = 1.0
            action = torch.tensor([act_val], dtype=torch.float32)

        obs, reward, done, truncated, info = env.step(action)
        state["obs_history"].append(obs)
        state["act_history"].append(action)
        state["ep_return"] += reward
        state["ep_steps"] += 1

        max_hist = cfg.policy_max_context + 2
        if len(state["obs_history"]) > max_hist:
            state["obs_history"] = state["obs_history"][-max_hist:]
            state["act_history"] = state["act_history"][-max_hist:]

        env_rgb = env.render_rgb(size=cfg.image_size)

        wm_rgb = env_rgb
        pred_rw = 0.0
        pred_val = 0.0
        if agent is not None and state["dream"] is not None:
            try:
                wm_rgb, pred_rw, pred_val = dream_step(
                    agent, state["dream"], action, cfg, device,
                )
            except Exception as e:
                logger.warning("WM dream step failed: %s", e)

        state["env_jpeg"] = numpy_to_jpeg_bytes(env_rgb)
        state["wm_jpeg"] = numpy_to_jpeg_bytes(wm_rgb)
        state["frame_id"] += 1

        mode = "POLICY" if state["use_policy"] else "KEYBOARD"
        payload = {
            "frame_id": state["frame_id"],
            "reward": float(reward),
            "pred_reward": float(pred_rw),
            "pred_value": float(pred_val),
            "ep_return": float(state["ep_return"]),
            "ep_steps": state["ep_steps"],
            "mode": mode,
            "done": bool(done),
        }

        if done:
            logger.info(
                "Episode done. Return=%.2f, Steps=%d",
                state["ep_return"], state["ep_steps"],
            )
            reset_env()
            state["frame_id"] += 1
            payload["frame_id"] = state["frame_id"]

        return payload

    @app.post("/step")
    async def step_endpoint(request: Request):
        body = await request.json()
        action_str = body.get("action", "none")
        return JSONResponse(process_action(action_str))

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        logger.info("WebSocket client connected")
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                action_str = msg.get("action", "none")
                payload = process_action(action_str)
                await ws.send_text(json.dumps(payload))
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Dreamer 4 — Web World Model Explorer",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default="config/cartpole_balance_light.yaml",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Run without checkpoint (untrained WM)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {}
    if args.device:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides=overrides)
    device = torch.device(cfg.device)

    agent: Dreamer4Agent | None = None
    has_checkpoint = False

    if args.checkpoint and not args.no_checkpoint:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        agent = Dreamer4Agent(cfg)
        tok = make_tokenizer(cfg)
        agent.set_tokenizer(tok)
        agent.update_prior_policy()
        agent = agent.to(device)

        meta = load_checkpoint(args.checkpoint, agent, map_location=cfg.device)
        logger.info(
            "Checkpoint loaded: train_step=%d, env_steps=%d",
            meta["train_step"], meta["env_steps"],
        )
        agent.eval()
        has_checkpoint = True

    elif not args.no_checkpoint:
        logger.info("No checkpoint — building untrained agent")
        agent = Dreamer4Agent(cfg)
        tok = make_tokenizer(cfg)
        agent.set_tokenizer(tok)
        agent = agent.to(device)
        agent.eval()

    else:
        logger.info("Running without agent (no WM predictions)")

    app = build_app(agent, cfg, device, has_checkpoint)

    logger.info("Starting server on http://%s:%d", args.host, args.port)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
