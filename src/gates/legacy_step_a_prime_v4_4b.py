# step_a_prime_v4_4b.py
# Fixed A'-0: symmetric B_U rows, offset in u dynamics.
# Rerun A'-0 / A'-1 / A'-2.

import numpy as np
import torch
import torch.nn as nn

# ---------------- v4.3 params (frozen) ----------------
A_SCALAR = 0.20
B_MAT = np.array([
    [-0.025, -0.010,  0.000,  0.015,  0.005],
    [-0.0075,-0.0225, 0.010,  0.000,  0.0125],
    [ 0.000, -0.0125,-0.020,  0.010,  0.005],
    [-0.010,  0.000, -0.0125, 0.0075,-0.0175],
], dtype=np.float64)
C_SCALAR = 0.40
D_VEC    = np.array([0.20]*4, dtype=np.float64)
SIGMA_X  = 0.035
RHO_Q    = 0.98
OFFSET_Q = 0.01
SIGMA_Q  = 0.020
SIGMA_OBS= 0.200

M_MAT = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
], dtype=np.float64)

THETA = 0.5; ALPHA = 8.0; GRID = 8

FULL_DIM  = 8
HIST_K    = 8
HIST_DIM  = FULL_DIM * HIST_K + 5 * HIST_K

# ---------------- u channel params ----------------
SIGMA_U   = 0.10
OFFSET_U  = 0.40
B_U       = B_MAT.copy()
B_U[1]    = B_U[0]                 # <-- symmetric rows (P1)
T_LOCK    = 5
T_EP      = T_LOCK + 1

RHO_U_SWEEP   = [0.10, 0.20, 0.40]
SIGMA_C_SWEEP = [0.005, 0.015, 0.030]
RHO_U_NOMINAL   = 0.20
SIGMA_C_NOMINAL = 0.005            # <-- updated based on A'-1

N_EPISODES = 5000


# ---------------- env ----------------
def run_episode(rho_u, sigma_c, seed):
    rng = np.random.RandomState(seed)

    q = rng.uniform(0.25, 0.75, 4)
    s = rng.uniform(0.25, 0.75, 4)
    w = rng.uniform(0.25, 0.75, 4)
    v = rng.uniform(0.25, 0.75, 4)
    u = rng.uniform(0.25, 0.75, 4)

    pos = (int(rng.randint(0, GRID)), int(rng.randint(0, GRID)))
    while True:
        gx = int(rng.randint(0, GRID)); gy = int(rng.randint(0, GRID))
        if (gx, gy) != pos:
            break
    goal = (gx, gy)

    signal_present = False
    signal_timer = int(rng.randint(20, 60))

    obs_list, act_list, cu_list = [], [], []
    u_pre_final = None

    y_q = np.clip(q + rng.normal(0, SIGMA_OBS, 4), 0, 1)
    x, yy = pos
    dx = (goal[0] - x) / GRID; dy = (goal[1] - yy) / GRID
    sig = 1.0 if signal_present else 0.0
    obs_list.append(np.concatenate([y_q, [dx, dy, 1.0, sig]]))
    cu_list.append(np.clip(u + rng.normal(0, sigma_c, 4), 0, 1))

    for t in range(T_EP - 1):
        a = int(rng.randint(0, 5))
        act_list.append(a)
        onehot = np.zeros(5); onehot[a] = 1.0
        q_t = q.copy()

        s = np.clip(D_VEC + A_SCALAR*s + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)
        w = np.clip(D_VEC + A_SCALAR*w + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)
        v = np.clip(D_VEC + A_SCALAR*v + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)

        u_pre = rho_u * u + OFFSET_U + B_U @ onehot + rng.normal(0, SIGMA_U, 4)
        u = np.clip(u_pre, 0, 1)
        u_pre_final = u_pre.copy()

        q = np.clip(RHO_Q * q_t + OFFSET_Q + rng.normal(0, SIGMA_Q, 4), 0, 1)

        signal_timer -= 1
        if signal_timer <= 0:
            signal_present = True

        y_q = np.clip(q + rng.normal(0, SIGMA_OBS, 4), 0, 1)
        x, yy = pos
        dx = (goal[0] - x) / GRID; dy = (goal[1] - yy) / GRID
        sig = 1.0 if signal_present else 0.0
        obs_list.append(np.concatenate([y_q, [dx, dy, 1.0, sig]]))
        cu_list.append(np.clip(u + rng.normal(0, sigma_c, 4), 0, 1))

    u_lock = u.copy()
    z = int(u_lock[0] > u_lock[1])
    z_pre = int(u_pre_final[0] > u_pre_final[1])

    return obs_list, act_list, cu_list, u_lock, u_pre_final, z, z_pre


def flatten_hist(obs_seq, act_seq):
    obs_seq = list(obs_seq); act_seq = list(act_seq)
    while len(obs_seq) < HIST_K:
        obs_seq = [np.zeros(FULL_DIM, dtype=np.float32)] + obs_seq
        act_seq = [4] + act_seq
    obs_seq = obs_seq[-HIST_K:]; act_seq = act_seq[-HIST_K:]
    a_oh = []
    for a in act_seq:
        v = np.zeros(5, dtype=np.float32); v[a] = 1.0
        a_oh.append(v)
    return np.concatenate([np.concatenate(obs_seq),
                           np.concatenate(a_oh)]).astype(np.float32)


def collect(rho_u, sigma_c, n_ep=N_EPISODES, base_seed=1000):
    H, CUR, CU, Z, ZPRE, ULOCK = [], [], [], [], [], []
    for ep in range(n_ep):
        seed = base_seed + ep + int(rho_u*1e4) + int(sigma_c*1e6)
        obs, act, cu, u_lock, u_pre_final, z, z_pre = run_episode(rho_u, sigma_c, seed)
        H.append(flatten_hist(obs, act))
        CUR.append(obs[-1].astype(np.float32))
        CU.append(cu[-1].astype(np.float32))
        Z.append(z); ZPRE.append(z_pre)
        ULOCK.append(u_lock)
    return (np.array(H, dtype=np.float32),
            np.array(CUR, dtype=np.float32),
            np.array(CU, dtype=np.float32),
            np.array(Z, dtype=np.int64),
            np.array(ZPRE, dtype=np.int64),
            np.array(ULOCK, dtype=np.float64))


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
    acc = float((pred == y_va).mean())
    baccs = []
    for c in [0, 1]:
        m = y_va == c
        if m.sum() > 0:
            baccs.append(float((pred[m] == c).mean()))
    return acc, float(np.mean(baccs)) if baccs else float("nan")


def eval_probe(X, y, model_ctor, n_folds=5, base_seed=0):
    n = X.shape[0]
    idx = np.random.RandomState(base_seed).permutation(n)
    folds = np.array_split(idx, n_folds)
    accs, baccs = [], []
    for k in range(n_folds):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        acc, bacc = train_probe(X[tr], y[tr], X[va], y[va],
                                model_ctor(X.shape[1]), seed=base_seed*100 + k)
        accs.append(acc); baccs.append(bacc)
    return float(np.mean(accs)), float(np.mean(baccs))


# ---------------- main ----------------
def main():
    print("=" * 78)
    print("STEP A' (v4.4b) — symmetric B_U, offset in u")
    print("=" * 78)
    print(f"T_LOCK = {T_LOCK},  SIGMA_U = {SIGMA_U},  OFFSET_U = {OFFSET_U}")
    print(f"B_U[1] = B_U[0]  (symmetric rows)")
    print(f"RHO_U sweep   = {RHO_U_SWEEP}   nominal = {RHO_U_NOMINAL}")
    print(f"SIGMA_C sweep = {SIGMA_C_SWEEP} nominal = {SIGMA_C_NOMINAL}")
    print()

    summary = []

    for rho_u in RHO_U_SWEEP:
        for sigma_c in SIGMA_C_SWEEP:
            H, CUR, CU, Z, ZPRE, ULOCK = collect(rho_u, sigma_c)

            p_z    = float(Z.mean())
            p_zpre = float(ZPRE.mean())
            m0, m1 = float(ULOCK[:,0].mean()), float(ULOCK[:,1].mean())
            s0, s1 = float(ULOCK[:,0].std()),  float(ULOCK[:,1].std())

            a0_pass = 0.45 <= p_z <= 0.55

            _, bacc_cu_lin = eval_probe(CU,  Z, LinProbe, base_seed=1)
            _, bacc_cu_mlp = eval_probe(CU,  Z, MLPProbe, base_seed=2)
            a1_cu_pass = max(bacc_cu_lin, bacc_cu_mlp) >= 0.95

            _, bacc_hist_lin = eval_probe(H, Z, LinProbe, base_seed=1)
            _, bacc_hist_mlp = eval_probe(H, Z, MLPProbe, base_seed=2)
            _, bacc_cur_lin  = eval_probe(CUR, Z, LinProbe, base_seed=1)
            _, bacc_cur_mlp  = eval_probe(CUR, Z, MLPProbe, base_seed=2)

            a2_hist_pass = max(bacc_hist_lin, bacc_hist_mlp) <= 0.85
            a2_cur_pass  = max(bacc_cur_lin,  bacc_cur_mlp)  <= 0.60

            print(f"--- rho_u={rho_u:.2f}  sigma_c={sigma_c:.3f} ---")
            print(f"  u stats:  mean[0]={m0:.4f} mean[1]={m1:.4f}  "
                  f"std[0]={s0:.4f} std[1]={s1:.4f}")
            print(f"  pre-clip: P(z_pre=1) = {p_zpre:.4f}  "
                  f"post-clip: P(z=1) = {p_z:.4f}   pass(A'-0)={a0_pass}")
            print(f"  A'-1  c_u -> z        = lin {bacc_cu_lin:.4f}  mlp {bacc_cu_mlp:.4f}"
                  f"   pass={a1_cu_pass}")
            print(f"  A'-2  H_full -> z     = lin {bacc_hist_lin:.4f}  mlp {bacc_hist_mlp:.4f}"
                  f"   pass={a2_hist_pass}")
            print(f"        CUR    -> z     = lin {bacc_cur_lin:.4f}  mlp {bacc_cur_mlp:.4f}"
                  f"   pass={a2_cur_pass}")
            print()

            summary.append((rho_u, sigma_c, p_z, p_zpre,
                            bacc_cu_lin, bacc_cu_mlp,
                            bacc_hist_lin, bacc_hist_mlp,
                            a0_pass, a1_cu_pass, a2_hist_pass, a2_cur_pass))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'rho_u':>6} {'sig_c':>7} {'P(z=1)':>8} {'A0':>4} {'A1':>4} {'A2h':>5} {'A2c':>5}")
    for s in summary:
        print(f"{s[0]:>6.2f} {s[1]:>7.3f} {s[2]:>8.4f} "
              f"{str(s[8]):>4} {str(s[9]):>4} {str(s[10]):>5} {str(s[11]):>5}")

    print("\nNOMINAL CONFIG (rho_u=%.2f, sigma_c=%.3f):" % (RHO_U_NOMINAL, SIGMA_C_NOMINAL))
    for s in summary:
        if s[0] == RHO_U_NOMINAL and abs(s[1] - SIGMA_C_NOMINAL) < 1e-6:
            print(f"  A'-0 pass: {s[8]}")
            print(f"  A'-1 pass: {s[9]}")
            print(f"  A'-2 (hist) pass: {s[10]}")
            print(f"  A'-2 (cur)  pass: {s[11]}")
            all_ok = s[8] and s[9] and s[10] and s[11]
            print(f"\n  NOMINAL ALL PASS: {all_ok}")
            if all_ok:
                print("  => Step A' gate cleared. Proceed to Step B' (head_U training).")
            else:
                print("  => Step A' not cleared on nominal config.")
            break


if __name__ == "__main__":
    main()