# v5_3_train.py
# Two-lock symmetric task, T_max=1.
# Variants: A, B, C, D, E, H.
# PPO training + post-training gates + TV matcher.
#
# ============================================================================
# PREREGISTERED SPECIFICATION (frozen before running)
# ============================================================================
# Environment:
#   T_MAX = 1, action = (a_u, a_w) ∈ {0,1}², N_ACTIONS = 4
#   z_u = 1[u_0[0] > u_0[1]],  z_w = 1[w_0[0] > w_0[1]]
#   reward: +2 if both correct, 0 if exactly one, -2 if none
#   private channels: c_x = clip(x_0 + N(0, SIGMA_C² I), 0, 1), SIGMA_C = 0.005
#
# Agent:
#   gru_vis: o_0 (8) → hv (24)
#   gru_priv: o_0 (8) → hp (8)      [variants B, C, D, H only]
#   head: [o_0, c_x, hp] (20) → aux (4), Sigmoid   [B, C, D, H]
#   actor: [hv, aux] (28) → 4 logits   [B, C, D, E, H]
#   actor: [hv] (24) → 4 logits        [A]
#   Variant targets: B→u_0, C→w_0, D→v_0, H→noise_target (independent)
#
# Training:
#   PPO (lr=3e-4, γ=0.99, λ=0.95, clip=0.2, ent=0.01, vf=0.5, grad_clip=0.5)
#   ROLLOUT_N=128, EPOCHS=4, LAMBDA_AUX=30
#   Budget: 200k env steps (episodes, since T=1)
#
# Preregistered tests:
#   R1, R2, R5a, R5b, R6 : Wilcoxon signed-rank, exact, two-sided, n=10, α=0.0125
#   R3, R4               : TOST equivalence, margin δ=0.15, α=0.05
#   No substitution between Wilcoxon and TOST after data collection.
#
# Sanity (bug-check only, NOT performance):
#   B, C, D on seed 0, 20k steps.
#   Checks: representation learned, no hv leak, D control ~chance,
#           TV matcher valid, no NaN.
#   Performance gates are POST-TRAINING only (see report).
#
# G4 (permutation control):
#   Status PENDING until executed on frozen checkpoints after sanity.
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, time
from scipy import stats

# ---------------- constants ----------------
GRID      = 8
SIGMA_OBS = 0.200
SIGMA_C   = 0.005
N_ACTIONS = 4

LR         = 3e-4
GAMMA      = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
GRAD_CLIP  = 0.5
LAMBDA_AUX = 30.0
ROLLOUT_N  = 128
EPOCHS     = 4

HIDDEN = 64
HV_DIM = 24
HP_DIM = 8

N_STEPS_TOTAL = 200_000
EVAL_EPISODES = 2000
TV_MIN_CAND   = 10
TV_QCAP       = 0.80
TV_SKIP_MAX   = 0.05

SEEDS    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
VARIANTS = ['A', 'B', 'C', 'D', 'E', 'H']

SANITY_SEED  = 0
SANITY_STEPS = 20_000

VARIANT_LABELS = {
    'A': 'A (no aux)',
    'B': 'B (u-channel aux)',
    'C': 'C (w-channel aux)',
    'D': 'D (v-channel aux, irrelevant)',
    'E': 'E (zero-aux control)',
    'H': 'H (noise target)',
}

CHECKPOINT_DIR = 'v5_3_checkpoints'


# ================= ENVIRONMENT =================
def reset_env(rng):
    """T=1. Returns dict."""
    q = rng.uniform(0.25, 0.75, 4).astype(np.float32)
    u = rng.uniform(0.25, 0.75, 4).astype(np.float32)
    w = rng.uniform(0.25, 0.75, 4).astype(np.float32)
    v = rng.uniform(0.25, 0.75, 4).astype(np.float32)

    y_q = np.clip(q + rng.normal(0, SIGMA_OBS, 4), 0, 1).astype(np.float32)

    pos = (int(rng.randint(0, GRID)), int(rng.randint(0, GRID)))
    while True:
        goal = (int(rng.randint(0, GRID)), int(rng.randint(0, GRID)))
        if goal != pos:
            break
    dx = (goal[0] - pos[0]) / GRID
    dy = (goal[1] - pos[1]) / GRID
    o = np.concatenate([y_q, [dx, dy, 1.0, 0.0]]).astype(np.float32)

    c_u = np.clip(u + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)
    c_w = np.clip(w + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)
    c_v = np.clip(v + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)

    noise_in  = rng.normal(0.5, 0.055, 4).astype(np.float32)
    noise_tgt = rng.normal(0.5, 0.055, 4).astype(np.float32)

    z_u = int(u[0] > u[1])
    z_w = int(w[0] > w[1])

    return {
        'o': o, 'c_u': c_u, 'c_w': c_w, 'c_v': c_v,
        'noise_in': noise_in, 'noise_tgt': noise_tgt,
        'u': u, 'w': w, 'v': v,
        'z_u': z_u, 'z_w': z_w,
    }


def compute_reward(a_idx, z_u, z_w):
    a_u = a_idx // 2
    a_w = a_idx % 2
    correct = (1 if a_u == z_u else 0) + (1 if a_w == z_w else 0)
    if correct == 2: return  2.0
    if correct == 1: return  0.0
    return -2.0


def variant_channel(variant, env):
    if variant == 'B': return env['c_u'], env['u']
    if variant == 'C': return env['c_w'], env['w']
    if variant == 'D': return env['c_v'], env['v']
    if variant == 'H': return env['noise_in'], env['noise_tgt']
    return None, None


# ================= AGENT =================
class Agent(nn.Module):
    def __init__(self, variant):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant

        self.fc_vis  = nn.Linear(8, HIDDEN)
        self.gru_vis = nn.GRU(HIDDEN, HV_DIM, batch_first=True)

        if variant in ('B', 'C', 'D', 'H'):
            self.fc_priv  = nn.Linear(8, HIDDEN)
            self.gru_priv = nn.GRU(HIDDEN, HP_DIM, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(8 + 4 + HP_DIM, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, 4), nn.Sigmoid())
        else:
            self.fc_priv  = None
            self.gru_priv = None
            self.head     = None

        actor_in = HV_DIM if variant == 'A' else HV_DIM + 4
        self.actor  = nn.Linear(actor_in, N_ACTIONS)
        self.critic = nn.Linear(actor_in, 1)

    def encode(self, o):
        x = F.relu(self.fc_vis(o)).unsqueeze(1)
        h, _ = self.gru_vis(x)
        return h[:, 0, :]

    def encode_private(self, o):
        x = F.relu(self.fc_priv(o)).unsqueeze(1)
        h, _ = self.gru_priv(x)
        return h[:, 0, :]

    def forward(self, o, c):
        hv = self.encode(o)
        if self.variant == 'A':
            aux = torch.zeros(o.shape[0], 4, device=o.device)
            cat = hv
        elif self.variant == 'E':
            aux = torch.zeros(o.shape[0], 4, device=o.device)
            cat = torch.cat([hv, aux], dim=-1)
        else:
            hp = self.encode_private(o)
            head_in = torch.cat([o, c, hp], dim=-1)
            aux = self.head(head_in)
            cat = torch.cat([hv, aux], dim=-1)
        logits = self.actor(cat)
        value = self.critic(cat).squeeze(-1)
        return {'logits': logits, 'value': value, 'aux': aux, 'hv': hv}


def sample_action(logits, deterministic=False):
    if deterministic:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


# ================= TRAINING =================
def ppo_step(agent, optimizer, batch, target_arr):
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

        logp_new = F.log_softmax(logits_new, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(logp_new - logp_old)

        L_pi = -torch.min(ratio * adv,
                          torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv).mean()
        L_v  = F.mse_loss(out['value'], returns)

        probs = F.softmax(logits_new, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()

        loss = L_pi + VF_COEF * L_v - ENT_COEF * entropy

        if agent.variant in ('B', 'C', 'D', 'H'):
            L_aux = F.mse_loss(out['aux'], target_arr)
            loss = loss + LAMBDA_AUX * L_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
        optimizer.step()


def train(variant, seed, n_steps, verbose=False, bootstrap_log=False):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)

    agent = Agent(variant)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)

    n_iter = n_steps // ROLLOUT_N
    for it in range(n_iter):
        env_list = [reset_env(rng) for _ in range(ROLLOUT_N)]

        o_arr = np.stack([e['o'] for e in env_list])
        c_list, tgt_list = [], []
        for e in env_list:
            c_in, t = variant_channel(variant, e)
            c_list.append(c_in if c_in is not None else np.zeros(4, np.float32))
            tgt_list.append(t if t is not None else np.zeros(4, np.float32))
        c_arr = np.stack(c_list).astype(np.float32)
        tgt_arr = np.stack(tgt_list).astype(np.float32)
        z_u_arr = np.array([e['z_u'] for e in env_list])
        z_w_arr = np.array([e['z_w'] for e in env_list])

        with torch.no_grad():
            o_t = torch.tensor(o_arr, dtype=torch.float32)
            c_t = torch.tensor(c_arr, dtype=torch.float32)
            out = agent(o_t, c_t)
            actions = sample_action(out['logits'])
            rewards = torch.tensor(
                [compute_reward(int(actions[i].item()), z_u_arr[i], z_w_arr[i])
                 for i in range(ROLLOUT_N)], dtype=torch.float32)

            if bootstrap_log and it == 0:
                delta_0 = rewards - out['value']
                print(f"    [T=1] bootstrap=0, delta_0 = r_0 - V(s_0), A_0 = delta_0")
                print(f"          mean delta_0 = {delta_0.mean().item():+.4f}")

        batch = {
            'o': o_t, 'c': c_t,
            'logits': out['logits'], 'action': actions,
            'reward': rewards, 'value': out['value'],
        }
        target_t = torch.tensor(tgt_arr, dtype=torch.float32)

        ppo_step(agent, opt, batch, target_t)

        if verbose and (it + 1) % 40 == 0:
            print(f"    iter {it+1}/{n_iter}  mean_r={rewards.mean().item():+.3f}")

    return agent


# ================= EVALUATION =================
def evaluate(agent, seed, n_episodes=EVAL_EPISODES):
    torch.manual_seed(seed + 1000); rng = np.random.RandomState(seed + 1000)
    agent.eval()

    o_l, c_l, aux_l, hv_l, action_l, logits_l = [], [], [], [], [], []
    z_u_l, z_w_l, reward_l = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            env = reset_env(rng)
            c_in, _ = variant_channel(agent.variant, env)
            o = torch.tensor(env['o']).unsqueeze(0)
            c = None if c_in is None else torch.tensor(c_in).unsqueeze(0)

            out = agent(o, c)
            a = sample_action(out['logits'])

            o_l.append(env['o'])
            c_l.append(c_in if c_in is not None else np.zeros(4, np.float32))
            aux_l.append(out['aux'].squeeze(0).numpy())
            hv_l.append(out['hv'].squeeze(0).numpy())
            action_l.append(int(a.item()))
            logits_l.append(out['logits'].squeeze(0).numpy())
            z_u_l.append(env['z_u']); z_w_l.append(env['z_w'])
            reward_l.append(compute_reward(int(a.item()), env['z_u'], env['z_w']))

    return {
        'o':       np.array(o_l),
        'c':       np.array(c_l),
        'aux':     np.array(aux_l),
        'hv':      np.array(hv_l),
        'action':  np.array(action_l),
        'logits':  np.array(logits_l),
        'z_u':     np.array(z_u_l),
        'z_w':     np.array(z_w_l),
        'reward':  np.array(reward_l),
    }


def bacc(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    accs = []
    for c in [0, 1]:
        m = target == c
        if m.sum() > 0:
            accs.append(float((pred[m] == c).mean()))
    return float(np.mean(accs))


class _ProbeMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


def _probe_classifier(X, y, seed=0, n_epochs=500):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n < 20: return 0.5
    idx = np.random.RandomState(seed).permutation(n)
    tr = idx[:int(n * 0.7)]; va = idx[int(n * 0.7):]
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    model = _ProbeMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(Xv).argmax(-1).numpy()
    return bacc(pred, y[va])


def gates_from_eval(variant, ev):
    g = {}
    if variant in ('B', 'C', 'D', 'H'):
        aux_sign = (ev['aux'][:, 0] > ev['aux'][:, 1]).astype(int)
        g['aux_u→z_u'] = bacc(aux_sign, ev['z_u'])
        g['aux_w→z_w'] = bacc(aux_sign, ev['z_w'])
    else:
        g['aux_u→z_u'] = 0.5
        g['aux_w→z_w'] = 0.5

    g['hv→z_u'] = _probe_classifier(ev['hv'], ev['z_u'], seed=42)
    g['hv→z_w'] = _probe_classifier(ev['hv'], ev['z_w'], seed=43)

    a_u = ev['action'] // 2
    a_w = ev['action'] % 2
    g['a_u→z_u'] = bacc(a_u, ev['z_u'])
    g['a_w→z_w'] = bacc(a_w, ev['z_w'])

    argmax_a = ev['logits'].argmax(axis=-1)
    am_u = argmax_a // 2
    am_w = argmax_a % 2
    g['argmax_a_u→z_u'] = bacc(am_u, ev['z_u'])
    g['argmax_a_w→z_w'] = bacc(am_w, ev['z_w'])
    return g


# ================= TV MATCHER =================
def compute_tv(ev):
    o  = ev['o']; hv = ev['hv']
    z_u = ev['z_u']; z_w = ev['z_w']
    pi  = F.softmax(torch.tensor(ev['logits'], dtype=torch.float32), dim=-1).numpy()

    X = np.concatenate([o, hv], axis=1).astype(np.float32)
    sq = (X ** 2).sum(1, keepdims=True)
    D = sq + sq.T - 2 * X @ X.T
    D = np.sqrt(np.clip(D, 0, None))
    np.fill_diagonal(D, np.inf)

    qcap = np.quantile(D[np.isfinite(D)], TV_QCAP)

    zu_i = z_u[:, None]; zu_j = z_u[None, :]
    zw_i = z_w[:, None]; zw_j = z_w[None, :]
    diff_mask = (zu_i != zu_j) | (zw_i != zw_j)
    dist_ok = D <= qcap
    cand = diff_mask & dist_ok
    n_cand = cand.sum(1)

    tv_vals = []
    skipped = 0
    for i in range(len(pi)):
        if n_cand[i] < TV_MIN_CAND:
            skipped += 1
            continue
        Dj = D[i].copy()
        Dj[~cand[i]] = np.inf
        j = int(np.argmin(Dj))
        tv_i = 0.5 * np.abs(pi[i] - pi[j]).sum()
        tv_vals.append(tv_i)

    frac_skipped = skipped / len(pi)
    return float(np.mean(tv_vals)) if tv_vals else float('nan'), frac_skipped


# ================= STATISTICS =================
def wilcoxon_two_sided(diffs):
    diffs = np.asarray(diffs, dtype=np.float64)
    if np.all(diffs == 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError:
        return float('nan'), float('nan')
    return float(stat), float(p)


def tost(diffs, margin):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    if se == 0: se = 1e-12
    t_low  = (mean + margin) / se
    t_high = (mean - margin) / se
    p_low  = 1 - stats.t.cdf(t_low,  df=n - 1)
    p_high = stats.t.cdf(t_high, df=n - 1)
    p = max(p_low, p_high)
    return float(p), bool(p < 0.05)


# ================= RUN =================
def run_one(variant, seed, n_steps, verbose=False, save_ckpt=True):
    t0 = time.time()
    agent = train(variant, seed, n_steps, verbose=verbose)
    ev = evaluate(agent, seed)
    g = gates_from_eval(variant, ev)
    tv_mean, tv_skip = compute_tv(ev)
    ret_mean = float(ev['reward'].mean())

    if save_ckpt:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f'{variant}_{seed}.pt')
        torch.save(agent.state_dict(), path)

    return {
        'variant': variant, 'seed': seed,
        'return_mean': ret_mean,
        'tv_mean': tv_mean, 'tv_skip_frac': tv_skip,
        'gates': g,
        'g4_status': 'PENDING',
        'time': time.time() - t0,
    }


def main():
    print("=" * 78)
    print("v5.3 PPO — two-lock symmetric task")
    print("=" * 78)
    print(f"T=1, N_ACTIONS={N_ACTIONS}, SIGMA_C={SIGMA_C}")
    print(f"Preregistered tests: R1/R2/R5/R6 Wilcoxon n=10 α=0.0125; "
          f"R3/R4 TOST δ=0.15 α=0.05")

    # ============ SANITY (bug-check only) ============
    print("\n" + "=" * 78)
    print("SANITY RUN (bug-check; B, C, D on seed 0, 20k)")
    print("=" * 78)
    print("  STOPPING RULE: bug-check only, NOT performance.")
    print("  Rationale: trace_b0.py showed 20k is too early for")
    print("             a_u_arg >= 0.85 on B; performance gates are")
    print("             POST-TRAINING only. No ckpt saved in sanity.\n")

    sanity_results = []
    for v in ['B', 'C', 'D']:
        print(f"  training {VARIANT_LABELS[v]} ...")
        r = run_one(v, SANITY_SEED, SANITY_STEPS, verbose=False, save_ckpt=False)
        sanity_results.append(r)
        print(f"  {VARIANT_LABELS[v]}:")
        print(f"    return_mean   = {r['return_mean']:+.4f}")
        print(f"    tv_mean       = {r['tv_mean']:.4f}   "
              f"skip={r['tv_skip_frac']:.4f}")
        print(f"    aux_u→z_u     = {r['gates']['aux_u→z_u']:.4f}")
        print(f"    aux_w→z_w     = {r['gates']['aux_w→z_w']:.4f}")
        print(f"    hv→z_u        = {r['gates']['hv→z_u']:.4f}")
        print(f"    hv→z_w        = {r['gates']['hv→z_w']:.4f}")

    b_s = next(r for r in sanity_results if r['variant'] == 'B')
    c_s = next(r for r in sanity_results if r['variant'] == 'C')
    d_s = next(r for r in sanity_results if r['variant'] == 'D')

    checks = {
        'B: aux_u→z_u ≥ 0.85 (head learned u)':
            b_s['gates']['aux_u→z_u'] >= 0.85,
        'B: hv→z_u ≤ 0.60 (no visual leak)':
            b_s['gates']['hv→z_u'] <= 0.60,
        'C: aux_w→z_w ≥ 0.85 (head learned w)':
            c_s['gates']['aux_w→z_w'] >= 0.85,
        'C: hv→z_w ≤ 0.60 (no visual leak)':
            c_s['gates']['hv→z_w'] <= 0.60,
        'D: aux_u→z_u ≤ 0.60 (D ignores u)':
            d_s['gates']['aux_u→z_u'] <= 0.60,
        'D: aux_w→z_w ≤ 0.60 (D ignores w)':
            d_s['gates']['aux_w→z_w'] <= 0.60,
        'TV matcher valid (max skip ≤ 0.05)':
            max(r['tv_skip_frac'] for r in sanity_results) <= TV_SKIP_MAX,
        'No NaN in returns':
            all(np.isfinite(r['return_mean']) for r in sanity_results),
    }

    print("\n  BUG-CHECK RESULTS:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    sanity_ok = all(checks.values())
    print(f"\n  SANITY (bug-check): {'PASS' if sanity_ok else 'FAIL'}")

    if not sanity_ok:
        print("  → STOP. Full run not performed. Diagnose first.")
        with open('v5_3_sanity.json', 'w') as f:
            json.dump({'sanity': sanity_results, 'checks': checks,
                       'sanity_ok': sanity_ok}, f, indent=2, default=str)
        return

    # ============ FULL RUN ============
    print("\n" + "=" * 78)
    print("FULL RUN (10 seeds × 6 variants × 200k)")
    print("=" * 78)

    all_results = []
    for variant in VARIANTS:
        print(f"\n  {VARIANT_LABELS[variant]}")
        for seed in SEEDS:
            r = run_one(variant, seed, N_STEPS_TOTAL, verbose=False, save_ckpt=True)
            all_results.append(r)
            print(f"    seed {seed}: ret={r['return_mean']:+.3f}  "
                  f"tv={r['tv_mean']:.4f}  "
                  f"aux→z_u={r['gates']['aux_u→z_u']:.3f}  "
                  f"aux→z_w={r['gates']['aux_w→z_w']:.3f}  "
                  f"a_u→z_u={r['gates']['argmax_a_u→z_u']:.3f}  "
                  f"a_w→z_w={r['gates']['argmax_a_w→z_w']:.3f}")

    # ============ AGGREGATE ============
    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS")
    print("=" * 78)

    by_variant = {v: [r for r in all_results if r['variant'] == v] for v in VARIANTS}
    returns = {v: np.array([r['return_mean'] for r in by_variant[v]]) for v in VARIANTS}
    tvs     = {v: np.array([r['tv_mean']     for r in by_variant[v]]) for v in VARIANTS}

    print("\nReturns (median [min, max] over 10 seeds):")
    for v in VARIANTS:
        print(f"  {VARIANT_LABELS[v]:35s}: "
              f"{np.median(returns[v]):+.3f} "
              f"[{returns[v].min():+.3f}, {returns[v].max():+.3f}]")

    print("\nTV (median [min, max]):")
    for v in VARIANTS:
        print(f"  {VARIANT_LABELS[v]:35s}: "
              f"{np.median(tvs[v]):.4f} "
              f"[{tvs[v].min():.4f}, {tvs[v].max():.4f}]")

    skip_fracs = np.array([r['tv_skip_frac'] for r in all_results])
    print(f"\nTV matcher skip fraction: median={np.median(skip_fracs):.4f}  "
          f"max={skip_fracs.max():.4f}")
    tv_gate_valid = bool(skip_fracs.max() <= TV_SKIP_MAX)
    print(f"  → TV gate {'valid' if tv_gate_valid else 'INVALID'}")

    # ============ PREREGISTERED TESTS ============
    print("\n" + "=" * 78)
    print("PREREGISTERED TESTS")
    print("=" * 78)
    print("  R1, R2, R5, R6: Wilcoxon signed-rank, exact, two-sided, α=0.0125")
    print("  R3, R4        : TOST equivalence, δ=0.15, α=0.05")
    print("  No substitution between Wilcoxon and TOST.\n")

    def report_wilcoxon(name, a, b):
        d = returns[a] - returns[b]
        stat, p = wilcoxon_two_sided(d)
        med = float(np.median(d))
        passed = bool(p < 0.0125 and med > 0)
        print(f"  {name}: median diff={med:+.3f}  p={p:.5f}  "
              f"{'PASS' if passed else 'FAIL'}")
        return {'median': med, 'p': p, 'pass': passed}

    r1  = report_wilcoxon("R1: return(B)−return(D)", 'B', 'D')
    r5a = report_wilcoxon("R5a: return(B)−return(A)", 'B', 'A')
    r5b = report_wilcoxon("R5b: return(B)−return(H)", 'B', 'H')
    r6  = report_wilcoxon("R6: return(B)−return(E)", 'B', 'E')

    d_ret = returns['B'] - returns['C']
    p_ret, pass_ret = tost(d_ret, margin=0.15)
    print(f"  R3: TOST |return(B)−return(C)| < 0.15  "
          f"median={np.median(d_ret):+.3f}  p={p_ret:.4f}  "
          f"{'PASS' if pass_ret else 'FAIL'}")

    d_tv = tvs['B'] - tvs['C']
    p_tv, pass_tv = tost(d_tv, margin=0.15)
    print(f"  R4: TOST |tv(B)−tv(C)| < 0.15          "
          f"median={np.median(d_tv):+.4f}  p={p_tv:.4f}  "
          f"{'PASS' if pass_tv else 'FAIL'}")

    d_tv_bd = tvs['B'] - tvs['D']
    stat2, p2 = wilcoxon_two_sided(d_tv_bd)
    med2 = float(np.median(d_tv_bd))
    pass2 = bool(p2 < 0.0125 and med2 > 0)
    print(f"  R2: TV(B)−TV(D)  median={med2:+.4f}  p={p2:.5f}  "
          f"{'PASS' if pass2 else 'FAIL'}")

    # ============ GATES SUMMARY ============
    print("\n" + "=" * 78)
    print("GATES (median over 10 seeds)")
    print("=" * 78)
    for v in VARIANTS:
        print(f"\n  {VARIANT_LABELS[v]}:")
        keys = by_variant[v][0]['gates'].keys()
        for k in keys:
            vals = np.array([r['gates'][k] for r in by_variant[v]])
            print(f"    {k}: median={np.median(vals):.4f} "
                  f"[{vals.min():.4f}, {vals.max():.4f}]")

    # ============ G4 STATUS ============
    print("\n" + "=" * 78)
    print("G4 STATUS")
    print("=" * 78)
    print("  G4 (permutation control): PENDING — not run yet.")
    print("  H1 analysis deferred until G4 executed on frozen checkpoints.")
    print(f"  Checkpoints saved to: ./{CHECKPOINT_DIR}/")
    print("  Until then, no PASS/FAIL on G4.")

    # ============ SAVE ============
    raw = {
        'config': {
            'SIGMA_C': SIGMA_C, 'N_STEPS_TOTAL': N_STEPS_TOTAL,
            'SEEDS': SEEDS, 'VARIANTS': VARIANTS,
            'LR': LR, 'LAMBDA_AUX': LAMBDA_AUX,
            'ROLLOUT_N': ROLLOUT_N, 'EPOCHS': EPOCHS,
            'T_MAX': 1, 'N_ACTIONS': N_ACTIONS,
        },
        'variant_labels': VARIANT_LABELS,
        'sanity': sanity_results,
        'sanity_ok': sanity_ok,
        'sanity_checks': checks,
        'all_results': all_results,
        'returns_by_variant': {v: returns[v].tolist() for v in VARIANTS},
        'tv_by_variant':      {v: tvs[v].tolist() for v in VARIANTS},
        'tv_matcher': {
            'state': ['o_0', 'hv_0'],
            'forbidden_same': ['z_u', 'z_w'],
            'distance_cap_q': TV_QCAP,
            'min_candidates': TV_MIN_CAND,
            'invalid_if_skip_frac_gt': TV_SKIP_MAX,
            'observed_max_skip_frac': float(skip_fracs.max()),
            'valid': tv_gate_valid,
        },
        'tests': {
            'R1':  r1,
            'R2':  {'median': med2, 'p': p2, 'pass': pass2},
            'R3':  {'median': float(np.median(d_ret)), 'p': p_ret, 'pass': pass_ret},
            'R4':  {'median': float(np.median(d_tv)),  'p': p_tv,  'pass': pass_tv},
            'R5a': r5a,
            'R5b': r5b,
            'R6':  r6,
        },
        'g4': {
            'status': 'PENDING',
            'note': 'To be run on frozen checkpoints after full run.',
            'checkpoints_dir': CHECKPOINT_DIR,
        },
    }
    with open('v5_3_results.json', 'w') as f:
        json.dump(raw, f, indent=2, default=str)
    print("\nSaved to v5_3_results.json")


if __name__ == '__main__':
    main()