# diagnose_b0.py
# Frozen checkpoint B_0.pt diagnostic.
# Loads v5_3_checkpoints/B_0.pt, evaluates on the same eval seed,
# and reports additional metrics not printed by the sanity gate.

import numpy as np
import torch
import torch.nn.functional as F

from v5_3_train import (
    Agent, reset_env, variant_channel, compute_reward, bacc,
    _probe_classifier, EVAL_EPISODES, CHECKPOINT_DIR,
    N_ACTIONS,
)

VARIANT   = 'B'
SEED      = 0
EVAL_SEED = SEED + 1000     # same as evaluate() uses
N_EP      = EVAL_EPISODES


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
        o_l.append(env['o'])
        c_l.append(c_in)
        z_u_l.append(env['z_u'])
        z_w_l.append(env['z_w'])
    return (np.stack(o_l).astype(np.float32),
            np.stack(c_l).astype(np.float32),
            np.array(z_u_l), np.array(z_w_l))


def forward(agent, o_t, c_t):
    with torch.no_grad():
        out = agent(o_t, c_t)
        probs = F.softmax(out['logits'], dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1)
        return {
            'logits':  out['logits'].numpy(),
            'probs':   probs.numpy(),
            'entropy': entropy.numpy(),
            'value':   out['value'].numpy(),
            'aux':     out['aux'].numpy(),
            'hv':      out['hv'].numpy(),
        }


def reward_of_action(a_np, z_u, z_w):
    return np.array([compute_reward(int(a), int(z_u[i]), int(z_w[i]))
                     for i, a in enumerate(a_np)])


def bacc_action(a_np, z, bit):
    pred = (a_np // 2) if bit == 0 else (a_np % 2)
    return bacc(pred, z)


def main():
    print("=" * 78)
    print(f"DIAGNOSTIC — frozen {VARIANT}_{SEED}.pt")
    print("=" * 78)
    print(f"eval seed = {EVAL_SEED},  n_ep = {N_EP}")
    print()

    agent = load_agent()
    o_arr, c_arr, z_u, z_w = collect_eval(N_EP)
    o_t = torch.tensor(o_arr)
    c_t = torch.tensor(c_arr)

    base = forward(agent, o_t, c_t)

    # counterfactual: shuffle c across episodes
    perm   = np.random.RandomState(EVAL_SEED + 999).permutation(N_EP)
    c_shuf = torch.tensor(c_arr[perm])
    shuf   = forward(agent, o_t, c_shuf)

    # counterfactual: zero c
    c_zero = torch.zeros_like(c_t)
    zero   = forward(agent, o_t, c_zero)

    # ============ POLICY ACTION ACCURACY ============
    print("--- POLICY ACTION ACCURACY ---")

    logits_t = torch.tensor(base['logits'])
    a_argmax = logits_t.argmax(-1).numpy()

    torch.manual_seed(EVAL_SEED)
    a_stoch = torch.multinomial(F.softmax(logits_t, dim=-1), 1).squeeze(-1).numpy()

    r_stoch  = reward_of_action(a_stoch,  z_u, z_w)
    r_argmax = reward_of_action(a_argmax, z_u, z_w)

    print(f"  return (stochastic)      = {r_stoch.mean():+.4f}")
    print(f"  return (argmax)          = {r_argmax.mean():+.4f}")
    print(f"  a_u→z_u (stochastic)     = {bacc_action(a_stoch,  z_u, 0):.4f}")
    print(f"  a_u→z_u (argmax)         = {bacc_action(a_argmax, z_u, 0):.4f}")
    print(f"  a_w→z_w (stochastic)     = {bacc_action(a_stoch,  z_w, 1):.4f}")
    print(f"  a_w→z_w (argmax)         = {bacc_action(a_argmax, z_w, 1):.4f}")

    # ============ REPRESENTATION ============
    print("\n--- REPRESENTATION ---")
    aux = base['aux']; hv = base['hv']
    aux_sign = (aux[:, 0] > aux[:, 1]).astype(int)
    print(f"  aux → z_u                = {bacc(aux_sign, z_u):.4f}")
    print(f"  aux → z_w                = {bacc(aux_sign, z_w):.4f}")
    print(f"  hv  → z_u (probe MLP)    = {_probe_classifier(hv, z_u, seed=42):.4f}")
    print(f"  hv  → z_w (probe MLP)    = {_probe_classifier(hv, z_w, seed=43):.4f}")

    # ============ ENTROPY / DISTRIBUTION ============
    print("\n--- POLICY DISTRIBUTION ---")
    probs = base['probs']
    ent   = base['entropy']
    print(f"  mean entropy             = {ent.mean():.4f}   "
          f"(uniform = {np.log(N_ACTIONS):.4f})")
    print(f"  mean max prob            = {probs.max(-1).mean():.4f}")
    print(f"  mean action probs        = "
          f"[{probs.mean(0)[0]:.4f}, {probs.mean(0)[1]:.4f}, "
          f"{probs.mean(0)[2]:.4f}, {probs.mean(0)[3]:.4f}]")
    hist = np.bincount(a_argmax, minlength=N_ACTIONS)
    print(f"  argmax histogram         = {hist.tolist()}")

    # cross-tab: does argmax correlate with z at all?
    print("\n  argmax vs (z_u, z_w):")
    for a in range(N_ACTIONS):
        m = a_argmax == a
        if m.sum() > 0:
            print(f"    a={a}: n={m.sum():4d}  "
                  f"P(z_u=1|a)={z_u[m].mean():.3f}  "
                  f"P(z_w=1|a)={z_w[m].mean():.3f}")

    # ============ COUNTERFACTUAL (aux intervention) ============
    print("\n--- COUNTERFACTUAL (c intervened, hv held fixed) ---")
    tv_shuf = 0.5 * np.abs(base['probs'] - shuf['probs']).sum(-1)
    tv_zero = 0.5 * np.abs(base['probs'] - zero['probs']).sum(-1)
    print(f"  TV(π_orig, π_shuf_c)  mean   = {tv_shuf.mean():.6f}")
    print(f"                        median = {np.median(tv_shuf):.6f}")
    print(f"                        max    = {tv_shuf.max():.6f}")
    print(f"                        frac>0.05 = {(tv_shuf > 0.05).mean():.4f}")
    print(f"  TV(π_orig, π_zero_c)  mean   = {tv_zero.mean():.6f}")

    # ============ CRITIC AUX-DEPENDENCE ============
    print("\n--- CRITIC AUX-DEPENDENCE ---")
    v_base = base['value']; v_shuf = shuf['value']; v_zero = zero['value']
    print(f"  value std                 = {v_base.std():.5f}")
    print(f"  |V_orig − V_shuf_c| mean  = {np.abs(v_base - v_shuf).mean():.5f}")
    print(f"  |V_orig − V_zero_c| mean  = {np.abs(v_base - v_zero).mean():.5f}")
    print(f"  corr(V_orig, V_shuf_c)    = "
          f"{np.corrcoef(v_base, v_shuf)[0,1]:.4f}")

    # ============ AUX SHUFFLE SANITY ============
    print("\n--- AUX SHUFFLE SANITY ---")
    aux_shuf = shuf['aux']
    aux_shuf_sign = (aux_shuf[:, 0] > aux_shuf[:, 1]).astype(int)
    print(f"  aux_shuf → z_u            = {bacc(aux_shuf_sign, z_u):.4f}  "
          f"(expected ≈ 0.50)")
    print(f"  aux_shuf → z_w            = {bacc(aux_shuf_sign, z_w):.4f}  "
          f"(expected ≈ 0.50)")

    # ============ VERDICT ============
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    aux_bacc     = bacc(aux_sign, z_u)
    argmax_bacc  = bacc_action(a_argmax, z_u, 0)
    stoch_bacc   = bacc_action(a_stoch,  z_u, 0)
    tv_mean      = float(tv_shuf.mean())
    value_dep    = float(np.abs(v_base - v_shuf).mean() / max(v_base.std(), 1e-9))

    print(f"  aux carries z:          aux→z_u   = {aux_bacc:.4f}")
    print(f"  argmax uses z:          a_u→z_u   = {argmax_bacc:.4f}")
    print(f"  stochastic uses z:      a_u→z_u   = {stoch_bacc:.4f}")
    print(f"  policy TV on c-shuffle:            = {tv_mean:.6f}")
    print(f"  critic aux-dependence:  |ΔV|/σ_V  = {value_dep:.4f}")
    print()

    if aux_bacc > 0.90 and argmax_bacc < 0.55 and stoch_bacc < 0.55 and tv_mean < 0.01:
        print("  CASE 1: Policy ignores aux.")
        print("          Representation ok, policy output unchanged by aux intervention.")
    elif aux_bacc > 0.90 and argmax_bacc > 0.65 and stoch_bacc < 0.60:
        print("  CASE 2: Policy uses aux at argmax; stochastic noise hides it.")
        print("          Consider lower entropy or argmax evaluation for this metric.")
    elif aux_bacc > 0.90 and tv_mean > 0.05 and argmax_bacc < 0.60:
        print("  CASE 3: Policy output differs on shuffled c, but argmax still chance.")
        print("          Investigate reward/action alignment or training dynamics.")
    else:
        print("  CASE 4: Ambiguous. See per-metric values above.")


if __name__ == '__main__':
    main()