# step_a_prime_v5_3.py
# Two-lock symmetric task, T_max=1.
# Pre-PPO gates A'-0 ... A'-4 on frozen v4.4b substrate.

import numpy as np
import torch
import torch.nn as nn
import json

GRID = 8
SIGMA_OBS = 0.200
SIGMA_C = 0.005

N_EPISODES = 5000
BASE_SEED  = 3000


# ---------------- env ----------------
def run_episode(seed):
    """T_max=1: reset, immediate decision."""
    rng = np.random.RandomState(seed)

    q = rng.uniform(0.25, 0.75, 4)
    u = rng.uniform(0.25, 0.75, 4)
    w = rng.uniform(0.25, 0.75, 4)
    v = rng.uniform(0.25, 0.75, 4)

    pos = (int(rng.randint(0, GRID)), int(rng.randint(0, GRID)))
    while True:
        gx = int(rng.randint(0, GRID))
        gy = int(rng.randint(0, GRID))
        if (gx, gy) != pos:
            break
    goal = (gx, gy)
    signal_present = False

    # o_0
    y_q = np.clip(q + rng.normal(0, SIGMA_OBS, 4), 0, 1)
    x, yy = pos
    dx = (goal[0] - x) / GRID
    dy = (goal[1] - yy) / GRID
    sig = 1.0 if signal_present else 0.0
    o_0 = np.concatenate([y_q, [dx, dy, 1.0, sig]]).astype(np.float32)

    # private channels
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


# ---------------- probes ----------------
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
        loss = nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
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


# ---------------- main ----------------
def main():
    print("=" * 78)
    print("STEP A' (v5.3) — two-lock symmetric task, T_max=1")
    print("=" * 78)
    print(f"SIGMA_C = {SIGMA_C},  N_EPISODES = {N_EPISODES}")
    print()

    O, CU, CW, CV, ZU, ZW = collect()

    # ---- A'-0: balance ----
    p_zu = float(ZU.mean())
    p_zw = float(ZW.mean())
    p_both = float(((ZU == 1) & (ZW == 1)).mean())
    a0_zu = 0.45 <= p_zu <= 0.55
    a0_zw = 0.45 <= p_zw <= 0.55
    a0 = a0_zu and a0_zw
    print(f"A'-0  P(z_u=1) = {p_zu:.4f}   pass={a0_zu}")
    print(f"      P(z_w=1) = {p_zw:.4f}   pass={a0_zw}")
    print(f"      P(z_u=1, z_w=1) = {p_both:.4f}   (indep: expect ~0.25)")
    print()

    # ---- A'-1: own channel ----
    b_uu, l_uu, m_uu = probe_max(CU, ZU)
    b_ww, l_ww, m_ww = probe_max(CW, ZW)
    a1 = (b_uu >= 0.95) and (b_ww >= 0.95)
    print(f"A'-1  c_u → z_u:  lin {l_uu:.4f}  mlp {m_uu:.4f}")
    print(f"      c_w → z_w:  lin {l_ww:.4f}  mlp {m_ww:.4f}")
    print(f"      pass (both ≥ 0.95) = {a1}")
    print()

    # ---- A'-2: cross-channel ----
    b_uw, l_uw, m_uw = probe_max(CU, ZW)
    b_wu, l_wu, m_wu = probe_max(CW, ZU)
    a2 = (b_uw <= 0.55) and (b_wu <= 0.55)
    print(f"A'-2  cross-channel:")
    print(f"      c_u → z_w:  lin {l_uw:.4f}  mlp {m_uw:.4f}")
    print(f"      c_w → z_u:  lin {l_wu:.4f}  mlp {m_wu:.4f}")
    print(f"      pass (both ≤ 0.55) = {a2}")
    print()

    # ---- A'-3: history (o_0) ----
    b_ou, l_ou, m_ou = probe_max(O, ZU)
    b_ow, l_ow, m_ow = probe_max(O, ZW)
    a3 = (b_ou <= 0.60) and (b_ow <= 0.60)
    print(f"A'-3  o_0 → targets (history check):")
    print(f"      o_0 → z_u:  lin {l_ou:.4f}  mlp {m_ou:.4f}")
    print(f"      o_0 → z_w:  lin {l_ow:.4f}  mlp {m_ow:.4f}")
    print(f"      pass (both ≤ 0.60) = {a3}")
    print()

    # ---- A'-4: c_v negative control ----
    b_vu, l_vu, m_vu = probe_max(CV, ZU)
    b_vw, l_vw, m_vw = probe_max(CV, ZW)
    a4 = (b_vu <= 0.55) and (b_vw <= 0.55)
    print(f"A'-4  c_v → targets (negative control):")
    print(f"      c_v → z_u:  lin {l_vu:.4f}  mlp {m_vu:.4f}")
    print(f"      c_v → z_w:  lin {l_vw:.4f}  mlp {m_vw:.4f}")
    print(f"      pass (both ≤ 0.55) = {a4}")
    print()

    print("=" * 78)
    print("GATE DECISION")
    print("=" * 78)
    all_pass = a0 and a1 and a2 and a3 and a4
    print(f"  A'-0 (balance):       {a0}")
    print(f"  A'-1 (own channel):   {a1}")
    print(f"  A'-2 (cross-channel): {a2}")
    print(f"  A'-3 (history):       {a3}")
    print(f"  A'-4 (c_v control):   {a4}")
    print(f"  ALL PASS: {all_pass}")
    if all_pass:
        print("  => Step A' v5.3 cleared. Ready for PPO spec.")
    else:
        print("  => Step A' v5.3 not cleared. Investigate before PPO.")

    with open("step_a_prime_v5_3_result.json", "w") as f:
        json.dump({
            "p_zu": p_zu, "p_zw": p_zw, "p_both": p_both,
            "bacc_cu_zu": b_uu, "bacc_cw_zw": b_ww,
            "bacc_cu_zw": b_uw, "bacc_cw_zu": b_wu,
            "bacc_o_zu":  b_ou, "bacc_o_zw":  b_ow,
            "bacc_cv_zu": b_vu, "bacc_cv_zw": b_vw,
            "gates": {"a0": a0, "a1": a1, "a2": a2,
                      "a3": a3, "a4": a4, "all": all_pass}
        }, f, indent=2)
    print("\nSaved to step_a_prime_v5_3_result.json")


if __name__ == "__main__":
    main()