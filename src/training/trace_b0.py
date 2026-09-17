# trace_b0.py
# Fresh 200k training of B, seed 0, with per-rollout logging.
# No hyperparameter changes. No architecture changes.
# Purpose: distinguish slow convergence from genuine optimization plateau.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from v5_3_train import (
    Agent, reset_env, variant_channel, compute_reward, bacc,
    N_ACTIONS, ROLLOUT_N, EPOCHS, LR, CLIP_EPS, ENT_COEF, VF_COEF,
    GRAD_CLIP, LAMBDA_AUX, VARIANT_LABELS,
)

VARIANT   = 'B'
SEED      = 0
N_STEPS   = 200_000
LOG_EVERY = 5           # every 5 rollouts = every 640 steps
EVAL_N    = 500         # eval episodes per log point


def sample_action(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def ppo_step_with_log(agent, optimizer, batch, target_t):
    logits_old = batch['logits']
    actions    = batch['action']
    rewards    = batch['reward']
    values_old = batch['value']
    o          = batch['o']
    c          = batch['c']

    adv_raw = (rewards - values_old).detach()
    adv_mean_raw = adv_raw.mean().item()
    adv_std_raw  = adv_raw.std().item()
    adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
    returns = rewards

    clip_fracs = []
    grad_norms = []

    for epoch in range(EPOCHS):
        out = agent(o, c)
        logits_new = out['logits']

        logp_new = F.log_softmax(logits_new, dim=-1).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        logp_old = F.log_softmax(logits_old, dim=-1).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(logp_new - logp_old)

        clipped_mask = (ratio < 1-CLIP_EPS) | (ratio > 1+CLIP_EPS)
        clip_fracs.append(clipped_mask.float().mean().item())

        L_pi = -torch.min(ratio * adv,
                          torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv).mean()
        L_v  = F.mse_loss(out['value'], returns)
        probs = F.softmax(logits_new, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()

        loss = L_pi + VF_COEF * L_v - ENT_COEF * entropy

        if agent.variant in ('B', 'C', 'D', 'H'):
            L_aux = F.mse_loss(out['aux'], target_t)
            loss = loss + LAMBDA_AUX * L_aux

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
        grad_norms.append(float(gn))
        optimizer.step()

    return {
        'adv_mean_raw':  adv_mean_raw,
        'adv_std_raw':   adv_std_raw,
        'clip_frac':     float(np.mean(clip_fracs)),
        'grad_norm_pre': float(np.mean(grad_norms)),
    }


def delta_logit_u(logits_np, z_u):
    dl = (logits_np[:, 2] + logits_np[:, 3]
          - logits_np[:, 0] - logits_np[:, 1])
    d0 = dl[z_u == 0].mean() if (z_u == 0).sum() > 0 else 0.0
    d1 = dl[z_u == 1].mean() if (z_u == 1).sum() > 0 else 0.0
    return float(d1 - d0)


def evaluate_quick(agent, seed, n_ep=EVAL_N):
    rng = np.random.RandomState(seed)
    o_l, c_l, z_u_l, z_w_l = [], [], [], []
    for _ in range(n_ep):
        env = reset_env(rng)
        c_in, _ = variant_channel(agent.variant, env)
        o_l.append(env['o']); c_l.append(c_in)
        z_u_l.append(env['z_u']); z_w_l.append(env['z_w'])

    o_t = torch.tensor(np.stack(o_l))
    c_t = torch.tensor(np.stack(c_l))
    z_u = np.array(z_u_l); z_w = np.array(z_w_l)

    was_training = agent.training
    agent.eval()
    with torch.no_grad():
        out = agent(o_t, c_t)
        logits_np = out['logits'].numpy()
        probs = F.softmax(out['logits'], dim=-1)
        aux = out['aux'].numpy()
        v = out['value'].numpy()

    a_arg = logits_np.argmax(-1)
    ret_arg = np.mean([compute_reward(int(a), int(z_u[i]), int(z_w[i]))
                       for i, a in enumerate(a_arg)])

    a_stoch = torch.multinomial(probs, 1).squeeze(-1).numpy()
    ret_stoch = np.mean([compute_reward(int(a), int(z_u[i]), int(z_w[i]))
                         for i, a in enumerate(a_stoch)])

    ent = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
    aux_bacc = bacc((aux[:, 0] > aux[:, 1]).astype(int), z_u)

    if was_training:
        agent.train()

    return {
        'arg_return':      float(ret_arg),
        'stoch_return':    float(ret_stoch),
        'arg_a_u_z_u':     bacc(a_arg // 2, z_u),
        'arg_a_w_z_w':     bacc(a_arg % 2, z_w),
        'stoch_a_u_z_u':   bacc(a_stoch // 2, z_u),
        'delta_logit_u':   delta_logit_u(logits_np, z_u),
        'entropy':         ent,
        'aux_to_z_u':      aux_bacc,
        'value_mean':      float(v.mean()),
        'value_std':       float(v.std()),
    }


def main():
    print("=" * 78)
    print(f"TRACE B_{SEED} — {N_STEPS} steps (fresh training)")
    print("=" * 78)
    print(f"variant={VARIANT}, seed={SEED}, ROLLOUT_N={ROLLOUT_N}, EPOCHS={EPOCHS}")
    print(f"lr={LR}, ent_coef={ENT_COEF}, lambda_aux={LAMBDA_AUX}, vf_coef={VF_COEF}")
    print(f"LOG_EVERY={LOG_EVERY} rollouts, EVAL_N={EVAL_N}")
    print()

    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    agent = Agent(VARIANT)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)

    n_iter = N_STEPS // ROLLOUT_N
    history = []

    for it in range(n_iter):
        env_list = [reset_env(rng) for _ in range(ROLLOUT_N)]
        o_arr   = np.stack([e['o'] for e in env_list]).astype(np.float32)
        c_list, tgt_list = [], []
        for e in env_list:
            c_in, t = variant_channel(VARIANT, e)
            c_list.append(c_in)
            tgt_list.append(t)
        c_arr   = np.stack(c_list).astype(np.float32)
        tgt_arr = np.stack(tgt_list).astype(np.float32)
        z_u_arr = np.array([e['z_u'] for e in env_list])
        z_w_arr = np.array([e['z_w'] for e in env_list])

        o_t   = torch.tensor(o_arr)
        c_t   = torch.tensor(c_arr)
        tgt_t = torch.tensor(tgt_arr)

        with torch.no_grad():
            out = agent(o_t, c_t)
            actions = sample_action(out['logits'])
            rewards = torch.tensor(
                [compute_reward(int(actions[i].item()), z_u_arr[i], z_w_arr[i])
                 for i in range(ROLLOUT_N)], dtype=torch.float32)

        batch = {
            'o': o_t, 'c': c_t,
            'logits': out['logits'], 'action': actions,
            'reward': rewards, 'value': out['value'],
        }

        step_log = ppo_step_with_log(agent, opt, batch, tgt_t)

        if (it + 1) % LOG_EVERY == 0 or it == 0 or it == n_iter - 1:
            ev = evaluate_quick(agent, seed=SEED + 10_000)
            row = {
                'iter':  it + 1,
                'steps': (it + 1) * ROLLOUT_N,
                **ev,
                **step_log,
            }
            history.append(row)
            print(f"  it {it+1:5d} ({row['steps']:>7d})  "
                  f"arg_ret={ev['arg_return']:+.3f}  "
                  f"stoch_ret={ev['stoch_return']:+.3f}  "
                  f"a_u_arg={ev['arg_a_u_z_u']:.3f}  "
                  f"a_u_stoch={ev['stoch_a_u_z_u']:.3f}  "
                  f"Δl_u={ev['delta_logit_u']:+.4f}  "
                  f"H={ev['entropy']:.3f}  "
                  f"aux→z_u={ev['aux_to_z_u']:.3f}  "
                  f"clip={step_log['clip_frac']:.3f}")

    with open('v5_3_trace_B0.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nSaved to v5_3_trace_B0.json")

    # --- final summary ---
    if history:
        first = history[0]
        last  = history[-1]
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  Δl_u:      {first['delta_logit_u']:+.4f} → {last['delta_logit_u']:+.4f}")
        print(f"  a_u_arg:   {first['arg_a_u_z_u']:.4f} → {last['arg_a_u_z_u']:.4f}")
        print(f"  arg_ret:   {first['arg_return']:+.4f} → {last['arg_return']:+.4f}")
        print(f"  entropy:   {first['entropy']:.4f} → {last['entropy']:.4f}")
        print(f"  value std: {first['value_std']:.4f} → {last['value_std']:.4f}")
        print(f"  clip:      {first['clip_frac']:.4f} → {last['clip_frac']:.4f}")

        # trajectory shape
        margins = np.array([r['delta_logit_u'] for r in history])
        n = len(margins)
        first_half = margins[:n//2].mean()
        second_half = margins[n//2:].mean()
        print(f"\n  mean Δl_u: first half={first_half:+.4f}  "
              f"second half={second_half:+.4f}  "
              f"trend={'GROWING' if second_half > first_half + 0.01 else 'FLAT/DECAYING'}")

        print("\n  DECISION RULE:")
        last_arg = last['arg_a_u_z_u']
        if last_arg >= 0.85:
            print("    → Sanity cutoff was too aggressive.")
            print("      At 200k, argmax a_u→z_u ≥ 0.85.")
            print("      Redefine sanity as bug-check only; use post-training gates.")
        elif last_arg >= 0.70:
            print("    → Partial convergence. Not a dead end, but marginal.")
            print("      Consider lower ent_coef (0.001) with justification,")
            print("      or 300k-400k budget, before opening full run.")
        else:
            print("    → Genuine optimization issue.")
            print("      Inspect L_pi vs L_v vs L_aux balance;")
            print("      check if critic gradient dominates shared representations.")


if __name__ == '__main__':
    main()