# step_a_prime_v5_3_sigma010.py
# v5.3 pre-PPO gates at SIGMA_C = 0.010.
# Env unchanged, only private-channel noise sigma differs.
# No PPO. Cheap filter before full run.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

GRID       = 8
SIGMA_OBS  = 0.200
SIGMA_C    = 0.010          # <-- only change
N_EPISODES = 5000
BASE_SEED  = 3000

RHO_U_NOMINAL = 0.20        # diagnostic env generation only; not used in training


def run_episode(seed):
    """T=1, two-lock env. Returns o_0, c_u, c_w, c_v, z_u, z_w."""
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
    o_0 = np.concatenate([y_q, [dx, dy, 1.0, 0.0]]).astype(np.float32)

    c_u = np.clip(u + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)
    c_w = np.clip(w + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)
    c_v = np.clip(v + rng.normal(0, SIGMA_C, 4), 0, 1).astype(np.float32)

    z_u = int(u[0] > u[1])
    z_w = int(w[0] > w[1])

    return o_0, c_u, c_w, c_v, z_u, z_w


def collect(n_ep=N_EPISODES, base_seed=BASE_SEED):
    O, CU, CW, CV = [], [], [], []
    ZU, ZW = [], []
    for ep in range(n_ep):
        o, cu, cw, cv, zu, zw = run_episode(base_seed + ep)
        O.append(o); CU.append(cu); CW.append(cw); CV.append(cv)
        ZU.append(zu); ZW.append(zw)
    return (np.array(O), np.array(CU), np.array(CW), np.array(CV),
            np.array(ZU, dtype=np.int64), np.array(ZW, dtype=np.int64))


class LinProbe(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 2)
    def forward(self, x): return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, d, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 2))
    def forward(self, x): return self.net(x)


def train_probe(X_tr, y_tr, X_va, y_va, model, seed, n_steps=3000, lr=1e-3):
    torch.manual_seed(seed)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = -1.0; best_state = None
    for step in range(n_steps):
        idx = torch.randperm(Xt.shape[0])[:256]
        loss = F.cross_entropy(model(Xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    baccs = []
    for c in [0, 1]:
        m = y_va == c
        if m.sum() > 0:
            baccs.append(float((pred[m] == c).mean()))
    return float((pred == y_va).mean()), float(np.mean(baccs)) if baccs else 0.5


def eval_probe(X, y, model_ctor, n_folds=5, base_seed=0):
    n = X.shape[0]
    idx = np.random.RandomState(base_seed).permutation(n)
    folds = np.array_split(idx, n_folds)
    baccs = []
    for k in range(n_folds):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        _, bacc = train_probe(X[tr], y[tr], X[va], y[va],
                              model_ctor(X.shape[1]), seed=base_seed*100 + k)
        baccs.append(bacc)
    return float(np.mean(baccs))


def probe_max(X, y):
    b_lin = eval_probe(X, y, LinProbe, base_seed=1)
    b_mlp = eval_probe(X, y, MLPProbe, base_seed=2)
    return max(b_lin, b_mlp), b_lin, b_mlp


def main():
    print("=" * 78)
    print("STEP A' (v5.3) — pre-PPO gates at SIGMA_C = 0.010")
    print("=" * 78)
    print(f"SIGMA_C = {SIGMA_C},  N_EPISODES = {N_EPISODES}")
    print()

    O, CU, CW, CV, ZU, ZW = collect()

    # A'-0: balance
    p_zu = float(ZU.mean())
    p_zw = float(ZW.mean())
    a0_zu = 0.45 <= p_zu <= 0.55
    a0_zw = 0.45 <= p_zw <= 0.55
    a0 = a0_zu and a0_zw
    print(f"A'-0  P(z_u=1) = {p_zu:.4f}   pass={a0_zu}")
    print(f"      P(z_w=1) = {p_zw:.4f}   pass={a0_zw}")

    # A'-1: own channel
    bu_lin, bu_mlp = eval_probe(CU, ZU, LinProbe, base_seed=1), eval_probe(CU, ZU, MLPProbe, base_seed=2)
    bw_lin, bw_mlp = eval_probe(CW, ZW, LinProbe, base_seed=1), eval_probe(CW, ZW, MLPProbe, base_seed=2)
    b_u = max(bu_lin, bu_mlp)
    b_w = max(bw_lin, bw_mlp)
    a1 = (b_u >= 0.95) and (b_w >= 0.95)
    print(f"\nA'-1  c_u → z_u:  lin {bu_lin:.4f}  mlp {bu_mlp:.4f}   (own={b_u:.4f})")
    print(f"      c_w → z_w:  lin {bw_lin:.4f}  mlp {bw_mlp:.4f}   (own={b_w:.4f})")
    print(f"      pass (max≥0.95 for both) = {a1}")

    # A'-2: cross-channel
    buw, buw_l, buw_m = probe_max(CU, ZW)
    bwu, bwu_l, bwu_m = probe_max(CW, ZU)
    a2 = (buw <= 0.55) and (bwu <= 0.55)
    print(f"\nA'-2  c_u → z_w:  lin {buw_l:.4f}  mlp {buw_m:.4f}   (max={buw:.4f})")
    print(f"      c_w → z_u:  lin {bwu_l:.4f}  mlp {bwu_m:.4f}   (max={bwu:.4f})")
    print(f"      pass (both ≤ 0.55) = {a2}")

    # A'-3: history
    bou, bou_l, bou_m = probe_max(O, ZU)
    bow, bow_l, bow_m = probe_max(O, ZW)
    a3 = (bou <= 0.60) and (bow <= 0.60)
    print(f"\nA'-3  o_0 → z_u:  lin {bou_l:.4f}  mlp {bou_m:.4f}   (max={bou:.4f})")
    print(f"      o_0 → z_w:  lin {bow_l:.4f}  mlp {bow_m:.4f}   (max={bow:.4f})")
    print(f"      pass (both ≤ 0.60) = {a3}")

    # A'-4: c_v control
    bvu, bvu_l, bvu_m = probe_max(CV, ZU)
    bvw, bvw_l, bvw_m = probe_max(CV, ZW)
    a4 = (bvu <= 0.55) and (bvw <= 0.55)
    print(f"\nA'-4  c_v → z_u:  lin {bvu_l:.4f}  mlp {bvu_m:.4f}   (max={bvu:.4f})")
    print(f"      c_v → z_w:  lin {bvw_l:.4f}  mlp {bvw_m:.4f}   (max={bvw:.4f})")
    print(f"      pass (both ≤ 0.55) = {a4}")

    # A'-5: information gap (new in v5.3 sigma010 check)
    gap_u = b_u - bou
    gap_w = b_w - bow
    a5 = (gap_u >= 0.30) and (gap_w >= 0.30)
    print(f"\nA'-5  gap (own channel − history):")
    print(f"      gap_u = {gap_u:+.4f}")
    print(f"      gap_w = {gap_w:+.4f}")
    print(f"      pass (both ≥ 0.30) = {a5}")

    print("\n" + "=" * 78)
    print("GATE DECISION")
    print("=" * 78)
    all_pass = a0 and a1 and a2 and a3 and a4 and a5
    print(f"  A'-0 (balance):       {a0}")
    print(f"  A'-1 (own channel):   {a1}")
    print(f"  A'-2 (cross-channel): {a2}")
    print(f"  A'-3 (history):       {a3}")
    print(f"  A'-4 (c_v control):   {a4}")
    print(f"  A'-5 (gap ≥ 0.30):    {a5}")
    print(f"  ALL PASS: {all_pass}")

    if all_pass:
        print("\n  => SIGMA_C = 0.010 A' cleared.")
        print("     Proceed to Step B' (head training) at this sigma.")
    else:
        print("\n  => SIGMA_C = 0.010 A' not cleared.")
        print("     H1 formulation remains in strong-SNR regime (0.005).")

    with open('step_a_prime_v5_3_sigma010_result.json', 'w') as f:
        json.dump({
            'SIGMA_C': SIGMA_C,
            'p_zu': p_zu, 'p_zw': p_zw,
            'bacc_cu_zu': b_u, 'bacc_cw_zw': b_w,
            'bacc_cu_zw': buw, 'bacc_cw_zu': bwu,
            'bacc_o_zu': bou, 'bacc_o_zw': bow,
            'bacc_cv_zu': bvu, 'bacc_cv_zw': bvw,
            'gap_u': gap_u, 'gap_w': gap_w,
            'gates': {'a0': a0, 'a1': a1, 'a2': a2,
                      'a3': a3, 'a4': a4, 'a5': a5, 'all': all_pass}
        }, f, indent=2)
    print("\nSaved to step_a_prime_v5_3_sigma010_result.json")


if __name__ == '__main__':
    main()