# step_b_prime_v5_3_sigma010.py
# Head-only training at SIGMA_C = 0.010, T=1, two-lock.
# No PPO, no actor. Verifies aux carries z at higher noise.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

GRID       = 8
SIGMA_OBS  = 0.200
SIGMA_C    = 0.010          # <-- only change
N_TRAIN    = 5000
N_VAL      = 2000
SEED       = 0
N_STEPS    = 6000
LR         = 1e-3
HIDDEN     = 64
HP_DIM     = 8
HV_DIM     = 24


def run_episode(seed):
    rng = np.random.RandomState(seed)
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

    z_u = int(u[0] > u[1])
    z_w = int(w[0] > w[1])
    return o, c_u, c_w, c_v, u, w, v, z_u, z_w


def collect(n_ep, base_seed):
    O, CU, CW, CV, U, W, V, ZU, ZW = [], [], [], [], [], [], [], [], []
    for ep in range(n_ep):
        o, cu, cw, cv, u, w, v, zu, zw = run_episode(base_seed + ep)
        O.append(o); CU.append(cu); CW.append(cw); CV.append(cv)
        U.append(u); W.append(w); V.append(v)
        ZU.append(zu); ZW.append(zw)
    return (np.stack(O), np.stack(CU), np.stack(CW), np.stack(CV),
            np.stack(U), np.stack(W), np.stack(V),
            np.array(ZU, dtype=np.int64), np.array(ZW, dtype=np.int64))


class Agent(nn.Module):
    """Same structure as v5.3 agent, but head used for isolated training."""
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.fc_priv  = nn.Linear(8, HIDDEN)
        self.gru_priv = nn.GRU(HIDDEN, HP_DIM, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(8 + 4 + HP_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 4), nn.Sigmoid())

    def forward(self, o, c):
        x = F.relu(self.fc_priv(o)).unsqueeze(1)
        hp, _ = self.gru_priv(x)
        hp = hp[:, 0, :]
        return self.head(torch.cat([o, c, hp], dim=-1))


def train_head(variant, O, C, target, n_steps=N_STEPS, seed=SEED, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    agent = Agent(variant)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    Ot = torch.tensor(O, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    Yt = torch.tensor(target, dtype=torch.float32)
    N = Ot.shape[0]

    for step in range(n_steps):
        idx = np.random.choice(N, 128, replace=False)
        out = agent(Ot[idx], Ct[idx])
        loss = F.mse_loss(out, Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 1000 == 0:
            print(f"    step {step:5d}  loss={loss.item():.5f}")
    return agent


def bacc(pred, target):
    accs = []
    for c in [0, 1]:
        m = target == c
        if m.sum() > 0:
            accs.append(float((pred[m] == c).mean()))
    return float(np.mean(accs))


def evaluate(agent, O, C, ZU, ZW, variant, shuffle_c=False, seed_eval=42):
    agent.eval()
    C_mod = C.copy()
    if shuffle_c:
        perm = np.random.RandomState(seed_eval).permutation(C.shape[0])
        C_mod = C_mod[perm]
    with torch.no_grad():
        aux = agent(torch.tensor(O, dtype=torch.float32),
                    torch.tensor(C_mod, dtype=torch.float32)).numpy()
    aux_sign = (aux[:, 0] > aux[:, 1]).astype(int)
    return {
        'aux_u_z_u': bacc(aux_sign, ZU),
        'aux_w_z_w': bacc(aux_sign, ZW),
    }


def main():
    print("=" * 78)
    print("STEP B' — head-only training at SIGMA_C = 0.010")
    print("=" * 78)
    print(f"N_TRAIN={N_TRAIN}, N_VAL={N_VAL}, N_STEPS={N_STEPS}\n")

    O_tr, CU_tr, CW_tr, CV_tr, U_tr, W_tr, V_tr, ZU_tr, ZW_tr = collect(N_TRAIN, 10000)
    O_va, CU_va, CW_va, CV_va, U_va, W_va, V_va, ZU_va, ZW_va = collect(N_VAL, 90000)

    results = {}

    # ----- B -----
    print("--- B (target = u_0) ---")
    agent_b = train_head('B', O_tr, CU_tr, U_tr, verbose=True)
    ev_b_normal  = evaluate(agent_b, O_va, CU_va, ZU_va, ZW_va, 'B')
    ev_b_shuffle = evaluate(agent_b, O_va, CU_va, ZU_va, ZW_va, 'B', shuffle_c=True)
    print(f"    normal:   aux→z_u = {ev_b_normal['aux_u_z_u']:.4f}   "
          f"aux→z_w = {ev_b_normal['aux_w_z_w']:.4f}")
    print(f"    shuffled: aux→z_u = {ev_b_shuffle['aux_u_z_u']:.4f}   "
          f"aux→z_w = {ev_b_shuffle['aux_w_z_w']:.4f}")
    results['B'] = {'normal': ev_b_normal, 'shuffle': ev_b_shuffle}

    # ----- C -----
    print("\n--- C (target = w_0) ---")
    agent_c = train_head('C', O_tr, CW_tr, W_tr, verbose=True)
    ev_c_normal  = evaluate(agent_c, O_va, CW_va, ZU_va, ZW_va, 'C')
    ev_c_shuffle = evaluate(agent_c, O_va, CW_va, ZU_va, ZW_va, 'C', shuffle_c=True)
    print(f"    normal:   aux→z_w = {ev_c_normal['aux_w_z_w']:.4f}   "
          f"aux→z_u = {ev_c_normal['aux_u_z_u']:.4f}")
    print(f"    shuffled: aux→z_w = {ev_c_shuffle['aux_w_z_w']:.4f}   "
          f"aux→z_u = {ev_c_shuffle['aux_u_z_u']:.4f}")
    results['C'] = {'normal': ev_c_normal, 'shuffle': ev_c_shuffle}

    # ----- D -----
    print("\n--- D (target = v_0, irrelevant) ---")
    agent_d = train_head('D', O_tr, CV_tr, V_tr, verbose=True)
    ev_d_normal = evaluate(agent_d, O_va, CV_va, ZU_va, ZW_va, 'D')
    print(f"    normal:   aux→z_u = {ev_d_normal['aux_u_z_u']:.4f}   "
          f"aux→z_w = {ev_d_normal['aux_w_z_w']:.4f}")
    results['D'] = {'normal': ev_d_normal}

    # ----- gates -----
    print("\n" + "=" * 78)
    print("GATES")
    print("=" * 78)
    checks = {
        'B: aux→z_u ≥ 0.90 (head learns u)':
            ev_b_normal['aux_u_z_u'] >= 0.90,
        'B: aux→z_w ≤ 0.60 (no leakage to w)':
            ev_b_normal['aux_w_z_w'] <= 0.60,
        'B: shuffle(c) → z_u ≤ 0.60 (causal)':
            ev_b_shuffle['aux_u_z_u'] <= 0.60,
        'C: aux→z_w ≥ 0.90 (head learns w)':
            ev_c_normal['aux_w_z_w'] >= 0.90,
        'C: aux→z_u ≤ 0.60 (no leakage to u)':
            ev_c_normal['aux_u_z_u'] <= 0.60,
        'C: shuffle(c) → z_w ≤ 0.60 (causal)':
            ev_c_shuffle['aux_w_z_w'] <= 0.60,
        'D: aux→z_u ≤ 0.60 (v irrelevant)':
            ev_d_normal['aux_u_z_u'] <= 0.60,
        'D: aux→z_w ≤ 0.60 (v irrelevant)':
            ev_d_normal['aux_w_z_w'] <= 0.60,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_pass = all(checks.values())
    print(f"\n  STEP B' (SIGMA_C=0.010): {'PASS' if all_pass else 'FAIL'}")

    if all_pass:
        print("\n  → Proceed to full 10-seed PPO at SIGMA_C = 0.010.")
    else:
        print("\n  → H1 remains fixed in strong-SNR regime (SIGMA_C=0.005).")

    with open('step_b_prime_v5_3_sigma010_result.json', 'w') as f:
        json.dump({'results': results, 'checks': checks, 'all_pass': all_pass},
                  f, indent=2, default=str)
    print("\nSaved to step_b_prime_v5_3_sigma010_result.json")


if __name__ == '__main__':
    main()