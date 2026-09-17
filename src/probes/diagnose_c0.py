# diagnose_c0.py
# Retrain C matching trace_c0.py RNG consumption, save checkpoint,
# then run focused diagnostic on frozen C_0.
#
# Answers:
#   - Is signal weak in probability space, or is Δlogit misleading?
#   - Is there systematic bias toward a_w=0?
#   - What is the 4-logit configuration by z_w?

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_3_train import (
    Agent, reset_env, variant_channel, compute_reward, bacc,
    N_ACTIONS, ROLLOUT_N, EPOCHS, LR, CLIP_EPS, ENT_COEF, VF_COEF,
    GRAD_CLIP, LAMBDA_AUX,
)

VARIANT   = 'C'
SEED      = 0
N_STEPS   = 200_000
LOG_EVERY = 5
EVAL_N    = 500
CKPT_PATH = 'v5_3_checkpoints/C_0.pt'

DIAG_N    = 5000
DIAG_SEED = 12345


def sample_action(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def ppo_step(agent, optimizer, batch, target_t):
    logits_old = batch['logits']
    actions    = batch['action']
    rewards    = batch['reward']
    values_old = batch['value']
    o          = batch['o']
    c          = batch['c']

    adv = (rewards - values_old).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    returns = rewards

    for _ in range(EPOCHS):
        out = agent(o, c)
        logits_new = out['logits']
        logp_new = F.log_softmax(logits_new, dim=-1).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        logp_old = F.log_softmax(logits_old, dim=-1).gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(logp_new - logp_old)

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
        torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
        optimizer.step()


def silent_eval_rng(agent, seed, n_ep):
    """Matches RNG consumption of trace_c0.py's evaluate_quick (multinomial)."""
    rng = np.random.RandomState(seed)
    o_l, c_l = [], []
    for _ in range(n_ep):
        env = reset_env(rng)
        c_in, _ = variant_channel(agent.variant, env)
        o_l.append(env['o']); c_l.append(c_in)
    o_t = torch.tensor(np.stack(o_l))
    c_t = torch.tensor(np.stack(c_l))
    was_training = agent.training
    agent.eval()
    with torch.no_grad():
        out = agent(o_t, c_t)
        probs = F.softmax(out['logits'], dim=-1)
        _ = torch.multinomial(probs, 1)     # same RNG-consuming op
    if was_training:
        agent.train()


def train_c(verbose=True):
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    agent = Agent(VARIANT)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    n_iter = N_STEPS // ROLLOUT_N

    for it in range(n_iter):
        env_list = [reset_env(rng) for _ in range(ROLLOUT_N)]
        o_arr = np.stack([e['o'] for e in env_list]).astype(np.float32)
        c_list, tgt_list = [], []
        for e in env_list:
            c_in, t = variant_channel(VARIANT, e)
            c_list.append(c_in); tgt_list.append(t)
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
        ppo_step(agent, opt, batch, tgt_t)

        if (it + 1) % LOG_EVERY == 0 or it == 0 or it == n_iter - 1:
            silent_eval_rng(agent, SEED + 10_000, EVAL_N)

        if verbose and ((it + 1) % 500 == 0 or it == n_iter - 1):
            print(f"  iter {it+1}/{n_iter}  mean_r_last_rollout={rewards.mean().item():+.4f}")

    return agent


def diagnostic(agent, n_ep=DIAG_N, seed=DIAG_SEED):
    print("\n" + "=" * 78)
    print("DIAGNOSTIC ON FROZEN C_0")
    print("=" * 78)
    print(f"n_ep={n_ep}, seed={seed}")

    torch.manual_seed(seed)
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

    a_arg = logits_np.argmax(-1)
    a_stoch = torch.multinomial(probs, 1).squeeze(-1).numpy()
    if was_training:
        agent.train()

    # --- 0. basics ---
    print("\n--- BASELINE ---")
    print(f"  P(z_w=1)              = {z_w.mean():.4f}")
    print(f"  aux -> z_w bacc       = {bacc((aux[:,0] > aux[:,1]).astype(int), z_w):.4f}")
    print(f"  argmax return         = "
          f"{np.mean([compute_reward(int(a), int(z_u[i]), int(z_w[i])) for i, a in enumerate(a_arg)]):+.4f}")
    print(f"  stoch return          = "
          f"{np.mean([compute_reward(int(a), int(z_u[i]), int(z_w[i])) for i, a in enumerate(a_stoch)]):+.4f}")

    # --- 1. Delta logit by z_w ---
    dl_w = (logits_np[:, 1] + logits_np[:, 3]
            - logits_np[:, 0] - logits_np[:, 2])
    print("\n--- Dlogit_w = (l1+l3) - (l0+l2) ---")
    print(f"  overall:  mean={dl_w.mean():+.4f}  std={dl_w.std():.4f}")
    for zw in [0, 1]:
        m = z_w == zw
        print(f"  z_w={zw}: n={m.sum():4d}  mean={dl_w[m].mean():+.4f}  "
              f"std={dl_w[m].std():.4f}")
    d0, d1 = dl_w[z_w==0].mean(), dl_w[z_w==1].mean()
    print(f"  gap (z=1 - z=0) = {d1-d0:+.4f}")

    # --- 2. P(a_w=1 | z_w) for argmax and stochastic ---
    for label, a in [('argmax', a_arg), ('stochastic', a_stoch)]:
        print(f"\n--- P(a_w=1 | z_w) [{label}] ---")
        p0 = ((a[z_w==0] % 2) == 1).mean()
        p1 = ((a[z_w==1] % 2) == 1).mean()
        print(f"  z_w=0: P(a_w=1) = {p0:.4f}")
        print(f"  z_w=1: P(a_w=1) = {p1:.4f}")
        print(f"  |P1-P0| = {abs(p1-p0):.4f}")
        print(f"  bacc = {bacc(a % 2, z_w):.4f}")

    # --- 3. Full 4-class distribution by z_w ---
    print("\n--- P(a=k | z_w) [argmax] ---")
    for zw in [0, 1]:
        m = z_w == zw
        print(f"  z_w={zw}  (n={m.sum()}):")
        for k in range(4):
            p = (a_arg[m] == k).mean()
            au, aw = k // 2, k % 2
            print(f"    a={k} (a_u={au},a_w={aw}): {p:.4f}")

    # --- 4. Confusion matrix a_w_arg vs z_w ---
    print("\n--- Confusion matrix (argmax) ---")
    print(f"                z_w=0    z_w=1")
    for aw_pred in [0, 1]:
        row = []
        for zw_true in [0, 1]:
            m = z_w == zw_true
            row.append(((a_arg[m] % 2) == aw_pred).mean())
        print(f"  a_w_pred={aw_pred}:   {row[0]:.4f}   {row[1]:.4f}")

    # --- 5. Mean logits by z_w ---
    print("\n--- Mean logits by z_w ---")
    for zw in [0, 1]:
        m = z_w == zw
        means = logits_np[m].mean(0)
        print(f"  z_w={zw}: [l0={means[0]:+.4f}, l1={means[1]:+.4f}, "
              f"l2={means[2]:+.4f}, l3={means[3]:+.4f}]")
    print("  overall: "
          f"[l0={logits_np[:,0].mean():+.4f}, "
          f"l1={logits_np[:,1].mean():+.4f}, "
          f"l2={logits_np[:,2].mean():+.4f}, "
          f"l3={logits_np[:,3].mean():+.4f}]")

    # --- 6. Bias check ---
    print("\n--- Bias check (argmax) ---")
    print(f"  P(a_w=1) overall = {(a_arg % 2).mean():.4f}")
    print(f"  P(a_u=1) overall = {(a_arg // 2).mean():.4f}")

    # --- 7. u-bit (expect chance) ---
    print("\n--- u-bit (expect ~0.50) ---")
    print(f"  argmax a_u -> z_u: {bacc(a_arg // 2, z_u):.4f}")
    print(f"  stoch  a_u -> z_u: {bacc(a_stoch // 2, z_u):.4f}")

    # --- 8. Conditional per-bit margins ---
    print("\n--- Per-bit conditional logit margins ---")
    # For a_w=1 to win, need max(l1, l3) > max(l0, l2).
    # Also useful: which pair among {l0,l1} vs {l2,l3} dominates.
    print(f"  mean max(l1,l3)      = {np.maximum(logits_np[:,1], logits_np[:,3]).mean():+.4f}")
    print(f"  mean max(l0,l2)      = {np.maximum(logits_np[:,0], logits_np[:,2]).mean():+.4f}")
    print(f"  mean (max(l1,l3) - max(l0,l2)) = "
          f"{(np.maximum(logits_np[:,1], logits_np[:,3]) - np.maximum(logits_np[:,0], logits_np[:,2])).mean():+.4f}")
    for zw in [0, 1]:
        m = z_w == zw
        print(f"  z_w={zw}: mean (max(l1,l3) - max(l0,l2)) = "
              f"{(np.maximum(logits_np[m,1], logits_np[m,3]) - np.maximum(logits_np[m,0], logits_np[m,2])).mean():+.4f}")


def main():
    print("=" * 78)
    print("DIAGNOSE C_0")
    print("=" * 78)

    if os.path.exists(CKPT_PATH):
        print(f"Loading checkpoint: {CKPT_PATH}")
        agent = Agent(VARIANT)
        agent.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    else:
        print(f"No checkpoint at {CKPT_PATH}.")
        print("Retraining C from scratch (RNG-matched to trace_c0.py).")
        print("This takes ~30-60 min. Progress every 500 iters.")
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        agent = train_c(verbose=True)
        torch.save(agent.state_dict(), CKPT_PATH)
        print(f"Saved checkpoint to {CKPT_PATH}")

    diagnostic(agent)


if __name__ == '__main__':
    main()