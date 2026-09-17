# diagnose_b0_v2.py
# Frozen B_0.pt diagnostic v2.
#
# Two corrections from v1:
#   - Section renamed: "ACTOR SENSITIVITY ∂logits/∂aux" (not "gradient of L_pi")
#   - Added real PPO L_pi gradient-flow test (exact replica of ppo_step first epoch)
#
# Sections:
#   1. Baseline forward
#   2. Actor sensitivity ∂logits/∂aux  (was mislabeled in v1)
#   3. Direct aux-flip intervention (swap aux[0] ↔ aux[1])
#   4. Full re-eval with flipped aux (a_u→z_u on flipped logits)
#   5. Conditional probabilities P(a_u=1 | z_u), P(a_w=1 | z_w)
#   6. Actor weights on hv vs aux positions
#   7. Real PPO L_pi gradient flow (backward through actual L_pi, aux detached)
#   8. Verdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_3_train import (
    Agent, reset_env, variant_channel, compute_reward, bacc,
    _probe_classifier, EVAL_EPISODES, CHECKPOINT_DIR, N_ACTIONS,
)

VARIANT   = 'B'
SEED      = 0
EVAL_SEED = SEED + 1000


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


def main():
    print("=" * 78)
    print(f"DIAGNOSTIC v2 — {VARIANT}_{SEED}.pt")
    print("=" * 78)

    agent = load_agent()
    o_arr, c_arr, z_u, z_w = collect_eval(EVAL_EPISODES)
    o_t = torch.tensor(o_arr)
    c_t = torch.tensor(c_arr)

    # ============ 1. BASELINE FORWARD ============
    agent.eval()
    with torch.no_grad():
        out = agent(o_t, c_t)
        logits_orig = out['logits'].clone()
        aux_orig    = out['aux'].clone()
        hv_orig     = out['hv'].clone()
    probs_orig = F.softmax(logits_orig, dim=-1)

    print("\n--- BASELINE ---")
    print(f"  aux → z_u                = "
          f"{bacc((aux_orig[:,0] > aux_orig[:,1]).numpy().astype(int), z_u):.4f}")
    print(f"  mean entropy             = "
          f"{(-(probs_orig * (probs_orig + 1e-9).log()).sum(-1)).mean().item():.4f}  "
          f"(uniform={np.log(N_ACTIONS):.4f})")
    print(f"  mean max prob            = {probs_orig.max(-1).values.mean().item():.4f}")

    # ============ 2. ACTOR SENSITIVITY: ∂logits / ∂aux ============
    # NOTE: this is NOT the gradient of L_pi.
    # It measures how much each actor logit responds to changes in aux.
    print("\n--- ACTOR SENSITIVITY: ∂logits / ∂aux ---")
    print("  (Jacobian of actor output w.r.t. aux — NOT L_pi gradient)")

    aux_in = aux_orig.clone().detach().requires_grad_(True)
    cat_in = torch.cat([hv_orig, aux_in], dim=-1)
    logits_g = agent.actor(cat_in)

    J = torch.zeros(4, 4)
    for a in range(4):
        grad_a = torch.autograd.grad(logits_g[:, a].sum(), aux_in,
                                     retain_graph=True, create_graph=False)[0]
        J[a] = grad_a.mean(0)

    print("  J (mean over eval set), rows=logit_a, cols=aux_i:")
    for a in range(4):
        print(f"    a={a}: [{J[a,0].item():+.4f}, {J[a,1].item():+.4f}, "
              f"{J[a,2].item():+.4f}, {J[a,3].item():+.4f}]")
    print(f"  Frobenius norm           = {J.norm().item():.4f}")

    # Direction corresponding to sign(aux[0] - aux[1]):
    d = torch.tensor([1.0, -1.0, 0.0, 0.0])
    Jd = J @ d
    print(f"  J·(e0−e1) = d logit_a / d(aux0−aux1):")
    for a in range(4):
        print(f"    a={a}: {Jd[a].item():+.4f}")
    print(f"  |J·(e0−e1)|              = {Jd.norm().item():.4f}")
    print("  (diagnostic scale; not a binary criterion)")

    # ============ 3. DIRECT AUX-FLIP ============
    print("\n--- DIRECT AUX-FLIP (swap aux[0] ↔ aux[1]) ---")
    aux_flip = aux_orig.clone()
    aux_flip[:, 0] = aux_orig[:, 1]
    aux_flip[:, 1] = aux_orig[:, 0]

    with torch.no_grad():
        cat_flip = torch.cat([hv_orig, aux_flip], dim=-1)
        logits_flip = agent.actor(cat_flip)
    probs_flip = F.softmax(logits_flip, dim=-1)

    tv_flip = 0.5 * (probs_orig - probs_flip).abs().sum(-1)
    print(f"  TV(π_orig, π_flipped)   mean   = {tv_flip.mean().item():.6f}")
    print(f"                          median = {np.median(tv_flip.numpy()):.6f}")
    print(f"                          max    = {tv_flip.max().item():.6f}")
    print(f"                          frac>0.01 = {(tv_flip > 0.01).float().mean().item():.4f}")
    print(f"                          frac>0.05 = {(tv_flip > 0.05).float().mean().item():.4f}")

    # ============ 4. FULL RE-EVAL WITH FLIPPED AUX ============
    a_orig = logits_orig.argmax(-1).numpy()
    a_flip = logits_flip.argmax(-1).numpy()

    def acc_u(a_np, z):
        return bacc(a_np // 2, z)

    def acc_w(a_np, z):
        return bacc(a_np % 2, z)

    print(f"\n  a_u→z_u argmax orig      = {acc_u(a_orig, z_u):.4f}")
    print(f"  a_u→z_u argmax flip      = {acc_u(a_flip, z_u):.4f}")
    print(f"  (flip should reduce accuracy if policy uses sign(aux))")

    # ============ 5. CONDITIONAL PROBABILITIES ============
    print("\n--- CONDITIONAL PROBABILITIES ---")
    p_au1_z0 = (probs_orig[z_u==0][:, 2] + probs_orig[z_u==0][:, 3]).mean().item()
    p_au1_z1 = (probs_orig[z_u==1][:, 2] + probs_orig[z_u==1][:, 3]).mean().item()
    print(f"  P(a_u=1 | z_u=0)         = {p_au1_z0:.4f}")
    print(f"  P(a_u=1 | z_u=1)         = {p_au1_z1:.4f}")
    print(f"  |Δ|                      = {abs(p_au1_z1 - p_au1_z0):.4f}")

    p_aw1_z0 = (probs_orig[z_w==0][:, 1] + probs_orig[z_w==0][:, 3]).mean().item()
    p_aw1_z1 = (probs_orig[z_w==1][:, 1] + probs_orig[z_w==1][:, 3]).mean().item()
    print(f"  P(a_w=1 | z_w=0)         = {p_aw1_z0:.4f}")
    print(f"  P(a_w=1 | z_w=1)         = {p_aw1_z1:.4f}")
    print(f"  |Δ|                      = {abs(p_aw1_z1 - p_aw1_z0):.4f}")

    # ============ 6. ACTOR WEIGHTS ============
    print("\n--- ACTOR WEIGHTS ---")
    W = agent.actor.weight.detach()
    W_aux = W[:, 24:28]
    W_hv  = W[:, :24]
    print(f"  |W_hv|  mean             = {W_hv.abs().mean().item():.4f}")
    print(f"  |W_hv|  max              = {W_hv.abs().max().item():.4f}")
    print(f"  |W_aux| mean             = {W_aux.abs().mean().item():.4f}")
    print(f"  |W_aux| max              = {W_aux.abs().max().item():.4f}")
    print(f"  ratio mean               = "
          f"{W_aux.abs().mean().item() / W_hv.abs().mean().item():.4f}")
    print(f"  W_aux block:")
    for a in range(4):
        print(f"    a={a}: [{W_aux[a,0].item():+.4f}, {W_aux[a,1].item():+.4f}, "
              f"{W_aux[a,2].item():+.4f}, {W_aux[a,3].item():+.4f}]")
    print(f"  bias                     = "
          f"{[round(b, 4) for b in agent.actor.bias.detach().tolist()]}")

    # ============ 7. REAL PPO L_pi GRADIENT FLOW ============
    # Exact replica of ppo_step first-epoch L_pi. Uses fresh rollout from
    # current policy (matching training-time sampling). aux and hv detached
    # so that we measure actor-parameter gradient only.
    print("\n--- REAL PPO L_pi GRADIENT FLOW ---")
    print("  (backward through actual L_pi; aux & hv detached; actor weights only)")

    torch.manual_seed(SEED + 500)
    rng = np.random.RandomState(SEED + 500)
    N = 2000

    o_l, c_l, z_u_l, z_w_l = [], [], [], []
    for _ in range(N):
        env = reset_env(rng)
        c_in, _ = variant_channel(VARIANT, env)
        o_l.append(env['o']); c_l.append(c_in)
        z_u_l.append(env['z_u']); z_w_l.append(env['z_w'])

    o_roll = torch.tensor(np.stack(o_l).astype(np.float32))
    c_roll = torch.tensor(np.stack(c_l).astype(np.float32))
    zu_roll = np.array(z_u_l); zw_roll = np.array(z_w_l)

    agent.eval()
    with torch.no_grad():
        out_r = agent(o_roll, c_roll)
        logits_r = out_r['logits']
        probs_r  = F.softmax(logits_r, dim=-1)
        actions  = torch.multinomial(probs_r, 1).squeeze(-1)
        rewards  = torch.tensor(
            [compute_reward(int(actions[i].item()), zu_roll[i], zw_roll[i])
             for i in range(N)], dtype=torch.float32)
        values   = out_r['value']

        # exact ppo_step advantage
        adv = (rewards - values).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        hv_r  = out_r['hv'].detach()
        aux_r = out_r['aux'].detach()

    # forward through actor only
    cat_r = torch.cat([hv_r, aux_r], dim=-1)
    logits_pi = agent.actor(cat_r)
    logp = F.log_softmax(logits_pi, dim=-1)
    logp_a = logp.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    L_pi = -(logp_a * adv).mean()

    agent.actor.zero_grad()
    if agent.actor.weight.grad is not None:
        agent.actor.weight.grad.zero_()
        agent.actor.bias.grad.zero_()
    L_pi.backward()

    g = agent.actor.weight.grad.detach()
    g_hv  = g[:, :24]
    g_aux = g[:, 24:28]
    g_bias = agent.actor.bias.grad.detach()

    print(f"  |grad actor hv|  mean    = {g_hv.abs().mean().item():.6e}")
    print(f"  |grad actor aux| mean    = {g_aux.abs().mean().item():.6e}")
    print(f"  |grad actor bias| mean   = {g_bias.abs().mean().item():.6e}")
    print(f"  ratio grad_aux / grad_hv = "
          f"{g_aux.abs().mean().item() / max(g_hv.abs().mean().item(), 1e-30):.4f}")

    # Also: gradient w.r.t. hv and aux inputs themselves
    hv_g = hv_r.clone().detach().requires_grad_(True)
    aux_g = aux_r.clone().detach().requires_grad_(True)
    cat_g = torch.cat([hv_g, aux_g], dim=-1)
    logits_gg = agent.actor(cat_g)
    logp_gg = F.log_softmax(logits_gg, dim=-1).gather(
        1, actions.unsqueeze(-1)).squeeze(-1)
    L_pi_g = -(logp_gg * adv).mean()
    g_hv_in, g_aux_in = torch.autograd.grad(
        L_pi_g, [hv_g, aux_g], retain_graph=False)
    print(f"  |∂L_pi/∂hv_input|  mean  = {g_hv_in.abs().mean().item():.6e}")
    print(f"  |∂L_pi/∂aux_input| mean  = {g_aux_in.abs().mean().item():.6e}")
    print(f"  ratio aux/hv (input)     = "
          f"{g_aux_in.abs().mean().item() / max(g_hv_in.abs().mean().item(), 1e-30):.4f}")

    # ============ 8. VERDICT ============
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    aux_bacc    = bacc((aux_orig[:,0] > aux_orig[:,1]).numpy().astype(int), z_u)
    tv_mean     = tv_flip.mean().item()
    delta_p     = abs(p_au1_z1 - p_au1_z0)
    ratio_w     = W_aux.abs().mean().item() / max(W_hv.abs().mean().item(), 1e-30)
    ratio_grad  = (g_aux.abs().mean().item()
                   / max(g_hv.abs().mean().item(), 1e-30))
    ratio_grad_in = (g_aux_in.abs().mean().item()
                     / max(g_hv_in.abs().mean().item(), 1e-30))

    print(f"  aux → z_u                         = {aux_bacc:.4f}")
    print(f"  TV on aux-flip                    = {tv_mean:.6f}")
    print(f"  |Δ P(a_u=1 | z_u)|                = {delta_p:.4f}")
    print(f"  |W_aux|/|W_hv|                    = {ratio_w:.4f}")
    print(f"  |grad_aux|/|grad_hv|  (weights)   = {ratio_grad:.4f}")
    print(f"  |∂L/∂aux|/|∂L/∂hv|    (inputs)    = {ratio_grad_in:.4f}")
    print()

    signals = 0
    if tv_mean > 0.02:          signals += 1
    if delta_p > 0.10:          signals += 1
    if ratio_w > 0.3:           signals += 1
    if ratio_grad_in > 0.3:     signals += 1

    print(f"  Dependent signals: {signals}/4")
    print()
    if signals <= 1:
        print("  CASE A: actor does not depend on aux.")
        print("          Representation ok, policy ignores it.")
    elif signals >= 3:
        print("  CASE B: actor structurally depends on aux.")
        print("          Weak behavioral accuracy is sampling noise, not")
        print("          absence of dependence. Entropy likely culprit.")
    else:
        print("  CASE C: partial dependence. See per-metric values.")

    print("\n  Architecture note:")
    print("    B target = u_0, so B cannot learn z_w by design.")
    print("    a_w→z_w ≈ 0.50 is consistent with specification, not a defect.")
    print("    Ceiling for B: return ≈ +1.")


if __name__ == '__main__':
    main()