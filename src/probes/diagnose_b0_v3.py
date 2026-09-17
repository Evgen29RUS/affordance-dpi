# diagnose_b0_v3.py
# Frozen B_0.pt — temperature sweep + Δlogit analysis.
# No training, no env changes. Read-only.

import numpy as np
import torch
import torch.nn.functional as F

from v5_3_train import (
    Agent, reset_env, variant_channel, compute_reward, bacc,
    EVAL_EPISODES, CHECKPOINT_DIR, N_ACTIONS,
)

VARIANT   = 'B'
SEED      = 0
EVAL_SEED = SEED + 1000

TEMPS     = [1.0, 0.5, 0.25, 0.1, 0.05]
N_SAMPLES = 5      # stochastic samples per T for a fair comparison


def load_agent():
    agent = Agent(VARIANT)
    path = f'{CHECKPOINT_DIR}/{VARIANT}_{SEED}.pt'
    agent.load_state_dict(torch.load(path, map_location='cpu'))
    agent.eval()
    return agent


def collect_eval(n_ep):
    rng = np.random.RandomState(EVAL_SEED)
    o_l, c_l, z_u_l, z_w_l = [], [], [], []
    for _ in range(n_ep):
        env = reset_env(rng)
        c_in, _ = variant_channel(VARIANT, env)
        o_l.append(env['o']); c_l.append(c_in)
        z_u_l.append(env['z_u']); z_w_l.append(env['z_w'])
    return (np.stack(o_l).astype(np.float32),
            np.stack(c_l).astype(np.float32),
            np.array(z_u_l), np.array(z_w_l))


def acc_bit(a_np, z, bit):
    pred = (a_np // 2) if bit == 0 else (a_np % 2)
    return bacc(pred, z)


def sample_with_seed(logits, T, seed, n_samples):
    """Sample n_samples actions per episode with the same seed."""
    torch.manual_seed(seed)
    probs = F.softmax(logits / T, dim=-1)
    # (N, n_samples)
    idx = torch.multinomial(probs, n_samples, replacement=True)
    return idx.numpy()


def main():
    print("=" * 78)
    print(f"DIAGNOSTIC v3 — {VARIANT}_{SEED}.pt")
    print("=" * 78)
    print(f"eval_seed={EVAL_SEED}, n_ep={EVAL_EPISODES}")
    print()

    agent = load_agent()
    o_arr, c_arr, z_u, z_w = collect_eval(EVAL_EPISODES)
    o_t = torch.tensor(o_arr)
    c_t = torch.tensor(c_arr)

    with torch.no_grad():
        out = agent(o_t, c_t)
        logits = out['logits']
    logits_np = logits.numpy()

    # ============ 1. Logit margin analysis ============
    print("--- LOGIT MARGIN ANALYSIS ---")
    # Δlogit_u = (l2 + l3) - (l0 + l1)  corresponds to log P(a_u=1)/P(a_u=0)
    # Δlogit_w = (l1 + l3) - (l0 + l2)  corresponds to log P(a_w=1)/P(a_w=0)
    dl_u = (logits_np[:, 2] + logits_np[:, 3] -
            logits_np[:, 0] - logits_np[:, 1])
    dl_w = (logits_np[:, 1] + logits_np[:, 3] -
            logits_np[:, 0] - logits_np[:, 2])

    for name, dl, z in [('Δlogit_u', dl_u, z_u), ('Δlogit_w', dl_w, z_w)]:
        print(f"\n  {name}:")
        print(f"    overall: mean={dl.mean():+.4f}  std={dl.std():.4f}")
        for zv in [0, 1]:
            m = z == zv
            print(f"    z={zv}:  n={m.sum():4d}  mean={dl[m].mean():+.4f}  "
                  f"std={dl[m].std():.4f}")
        d0 = dl[z == 0].mean()
        d1 = dl[z == 1].mean()
        pooled_std = np.sqrt((dl[z==0].var() + dl[z==1].var()) / 2)
        cohens_d = (d1 - d0) / max(pooled_std, 1e-9)
        print(f"    mean(z=1) − mean(z=0) = {d1 - d0:+.4f}")
        print(f"    Cohen's d              = {cohens_d:+.4f}")

    # ============ 2. Temperature sweep ============
    print("\n" + "=" * 78)
    print("TEMPERATURE SWEEP")
    print("=" * 78)
    print(f"T values: {TEMPS},  n_samples per T: {N_SAMPLES}")
    print()

    print(f"  {'T':>6} {'entropy':>8} {'max_prob':>9} "
          f"{'return':>9} {'a_u→z_u':>8} {'a_w→z_w':>8} "
          f"{'argmax_ret':>11}")
    print("  " + "-" * 74)

    # argmax (deterministic) as reference
    a_arg = logits_np.argmax(-1)
    ret_arg = np.array([compute_reward(int(a), int(z_u[i]), int(z_w[i]))
                        for i, a in enumerate(a_arg)])
    arg_u = acc_bit(a_arg, z_u, 0)
    arg_w = acc_bit(a_arg, z_w, 1)
    ent_arg = -(np.exp(logits_np) / np.exp(logits_np).sum(-1, keepdims=True)
                * (logits_np - np.log(np.exp(logits_np).sum(-1, keepdims=True)))
                ).sum(-1)
    # use straightforward softmax entropy
    p_arg = F.softmax(logits, dim=-1).numpy()
    ent_arg = -(p_arg * np.log(p_arg + 1e-12)).sum(-1)
    print(f"  {'argmax':>6} {ent_arg.mean():>8.4f} {p_arg.max(-1).mean():>9.4f} "
          f"{ret_arg.mean():>+9.4f} {arg_u:>8.4f} {arg_w:>8.4f} "
          f"{'—':>11}")

    results = []
    for T in TEMPS:
        # entropy of the temperature-adjusted distribution
        p_T = F.softmax(logits / T, dim=-1)
        ent_T = (-(p_T * (p_T + 1e-12).log()).sum(-1)).mean().item()
        maxp_T = p_T.max(-1).values.mean().item()

        # sample N_SAMPLES per episode, average metrics
        idx = sample_with_seed(logits, T, seed=SEED*7 + int(T*100), n_samples=N_SAMPLES)
        # idx shape (N, n_samples)
        rets, au_accs, aw_accs = [], [], []
        for s in range(N_SAMPLES):
            a = idx[:, s]
            rets.append(np.array([compute_reward(int(a[i]), int(z_u[i]), int(z_w[i]))
                                  for i in range(len(a))]))
            au_accs.append(acc_bit(a, z_u, 0))
            aw_accs.append(acc_bit(a, z_w, 1))
        ret_mean = np.mean([r.mean() for r in rets])
        au_mean  = np.mean(au_accs)
        aw_mean  = np.mean(aw_accs)
        ret_std  = np.std([r.mean() for r in rets])

        print(f"  {T:>6.2f} {ent_T:>8.4f} {maxp_T:>9.4f} "
              f"{ret_mean:>+9.4f} {au_mean:>8.4f} {aw_mean:>8.4f} "
              f"{'':>11}")

        results.append({
            'T': T, 'entropy': ent_T, 'max_prob': maxp_T,
            'return': ret_mean, 'return_std': ret_std,
            'a_u_z_u': au_mean, 'a_w_z_w': aw_mean,
        })

    # ============ 3. Interpretation ============
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    r_T1   = next(r for r in results if r['T'] == 1.0)
    r_Tsm  = next(r for r in results if r['T'] == TEMPS[-1])

    print(f"  return T=1.00          = {r_T1['return']:+.4f}")
    print(f"  return T={TEMPS[-1]:.2f}          = {r_Tsm['return']:+.4f}")
    print(f"  return argmax          = {ret_arg.mean():+.4f}")
    print(f"  argmax  a_u→z_u        = {arg_u:.4f}")
    print()

    # key diagnostic questions
    gain_T   = r_Tsm['return'] - r_T1['return']
    ceiling  = ret_arg.mean() - r_T1['return']

    print(f"  Gain from lowering T:  {gain_T:+.4f}")
    print(f"  Argmax ceiling:        {ceiling:+.4f}")
    print()

    if r_Tsm['return'] > ret_arg.mean() - 0.05:
        print("  → Temperature fixes stochastic sampling.")
        print("    Logits already contain the signal.")
        print("    At high T, argmax plateaus at "
              f"{ret_arg.mean():+.3f}; that is the current policy ceiling.")
    elif gain_T > 0.05:
        print("  → Temperature helps but does not close the gap to argmax.")
        print("    Both sampling noise AND weak logit margin are factors.")
    else:
        print("  → Temperature does NOT help. The issue is in the logit margin.")
        print("    Even at T→0, policy is bounded by argmax performance.")

    # Is argmax itself weak?
    print()
    if arg_u < 0.80:
        print(f"  → Even argmax a_u→z_u = {arg_u:.4f} is far from 0.99.")
        print("    The policy did NOT fully learn the aux→action mapping.")
        print("    Entropy is not the main cause; the learned dependence is weak.")
    else:
        print(f"  → argmax a_u→z_u = {arg_u:.4f} is high.")
        print("    Weak stochastic performance is a sampling/temperature issue.")

    # ============ 4. Where does the reward come from? ============
    print("\n" + "=" * 78)
    print("REWARD DECOMPOSITION (argmax)")
    print("=" * 78)
    au_c = (a_arg // 2 == z_u)
    aw_c = (a_arg % 2 == z_w)
    print(f"  P(a_u correct)  = {au_c.mean():.4f}")
    print(f"  P(a_w correct)  = {aw_c.mean():.4f}")
    print(f"  P(both correct) = {(au_c & aw_c).mean():.4f}")
    print(f"  P(only u)       = {(au_c & ~aw_c).mean():.4f}")
    print(f"  P(only w)       = {(~au_c & aw_c).mean():.4f}")
    print(f"  P(neither)      = {(~au_c & ~aw_c).mean():.4f}")
    print(f"  E[R] check      = "
          f"{2*(au_c & aw_c).mean() - 2*(~au_c & ~aw_c).mean():+.4f}  "
          f"(observed argmax return = {ret_arg.mean():+.4f})")


if __name__ == '__main__':
    main()