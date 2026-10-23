# AlohaPick: Dreamer 4 world model + RL, the general recipe

Vision AlohaPick (MuJoCo Playground fork; cameras `view_0` + `view_2`, 128×256; 14-D actions; 120-step episodes;
dense reward ~1.09 per step while the box is held). Results on 512 held-out episodes (seeds 900000+).

## The recipe

1. **Data, tokenizer, world model, clone** (phases 1–2): `collect_aloha_npy.py` (2 000 scripted-expert episodes with
   speed / noise dials), tokenizer and bootstrapped world model as in the obstacle recipe, phase 2 cloning
   (`train_heads --action_bins 41 --dyn_weight 10 --d_hidden 1024 --fwd_weight 0 --value_bin_high 6.5`).
2. **Online loop** (`dyna_loop.sh`, ~10 iterations of ~11 min): PMPO in imagination with **32-step dreams** (dreams
   stay faithful for ~40 steps; 120-step dreams turn failures into successes) → **128 real episodes** with the
   current policy → **refresh**: the transformer keeps training on all data with 50 % of its windows and of the dream
   starts from the online collections (`--dyn_online_frac`, `--start_online_frac`), the policy clones the top 25 % of
   all real episodes by return (`--bc_online_top`), and every 50th PMPO update learns from the newest real episodes
   (`--real_data`, `--real_every 50`; 5 collapses the policy onto one batch). Gentle refresh rates (1e-4 / 5e-5, 100-step
   ramp), or the policy breaks.
3. **Pick** the iteration with the best collected return (iterations swing by ±0.1 success).
4. **Real-env PMPO polish** (`train_pmpo --real_env 32`), ~1 000 updates; a continuation with `--real_env 64 --lr 3e-4`
   adds a few points.

`recipe.sh <gpu> <tag> <clone ckpt> <world-model run>` runs 2–4 and the held-out evaluation (`heldout_queue.sh`).

| held-out | greedy success / return | sampled |
|---|---|---|
| clone, 500 cloning steps | 0.60 / 41.7 | 0.80 / 49.4 |
| → online loop (1 024 real episodes) | 0.79 / 80.7 | 0.78 / 79.5 |
| → polish (+32 000) | 0.97 / 96.1 | 0.97 / 95.9 |
| → polish continued (64 / update, LR 3e-4) | 0.95 / 98.9 | **0.98 / 99.4** |
| 2nd seed of the same pipeline | 0.97 / 98.3 | 0.96 / 97.6 |
| for scale: imagination only (horizon 32) | 0.91 / 72.7 | 0.90 / 72.4 |
| for scale: real-env PMPO only (25 600 episodes) | 0.99 / 73.1 | 0.99 / 73.4 |

The BC + PPO reference is ~104. Other tools: `collect_policy.py` (episodes of a trained policy), `wm_replay.py` (does
the world model predict failure? recorded actions replayed through the dream loop), `collect_aloha_div.py` (diverse
scripted episodes — did not make dreams more faithful to policy data).
