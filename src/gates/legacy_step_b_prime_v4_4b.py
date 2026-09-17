# step_b_prime_v4_4b.py
# Train head_U on v4.4b data. Diagnostic only, no PPO.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

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

GRID = 8
THETA = 0.5
ALPHA = 8.0

# ---------------- u channel ----------------
SIGMA_U  = 0.10
OFFSET_U = 0.40
B_U = B_MAT.copy()
B_U[1] = B_U[0]
T_LOCK = 5
T_EP   = T_LOCK + 1   # 6

RHO_U_NOMINAL   = 0.20
SIGMA_C_NOMINAL = 0.005

N_TRAIN = 4000
N_VAL   = 1000


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

    obs_seq, cu_seq, u_seq = [], [], []

    def emit(q, u, pos, goal, signal_present, rng, sigma_c):
        y_q = np.clip(q + rng.normal(0, SIGMA_OBS, 4), 0, 1)
        x, yy = pos
        dx = (goal[0] - x) / GRID
        dy = (goal[1] - yy) / GRID
        sig = 1.0 if signal_present else 0.0
        return (np.concatenate([y_q, [dx, dy, 1.0, sig]]),
                np.clip(u + rng.normal(0, sigma_c, 4), 0, 1))

    o_t, c_t = emit(q, u, pos, goal, signal_present, rng, sigma_c)
    obs_seq.append(o_t); cu_seq.append(c_t); u_seq.append(u.copy())

    for t in range(T_EP - 1):
        a = int(rng.randint(0, 5))
        onehot = np.zeros(5); onehot[a] = 1.0
        q_t = q.copy()

        s = np.clip(D_VEC + A_SCALAR*s + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)
        w = np.clip(D_VEC + A_SCALAR*w + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)
        v = np.clip(D_VEC + A_SCALAR*v + B_MAT@onehot + C_SCALAR*q_t
                    + rng.normal(0, SIGMA_X, 4), 0, 1)
        u = np.clip(rho_u*u + OFFSET_U + B_U@onehot
                    + rng.normal(0, SIGMA_U, 4), 0, 1)
        q = np.clip(RHO_Q*q_t + OFFSET_Q + rng.normal(0, SIGMA_Q, 4), 0, 1)

        signal_timer -= 1
        if signal_timer <= 0:
            signal_present = True

        o_t, c_t = emit(q, u, pos, goal, signal_present, rng, sigma_c)
        obs_seq.append(o_t); cu_seq.append(c_t); u_seq.append(u.copy())

    z = int(u_seq[-1][0] > u_seq[-1][1])

    return (np.array(obs_seq, dtype=np.float32),
            np.array(cu_seq, dtype=np.float32),
            np.array(u_seq, dtype=np.float32),
            z)


def collect_data(n_ep, rho_u, sigma_c, base_seed=2000):
    O, C, U, Z = [], [], [], []
    for ep in range(n_ep):
        seed = base_seed + ep + int(rho_u*1e4) + int(sigma_c*1e6)
        o, c, u, z = run_episode(rho_u, sigma_c, seed)
        O.append(o); C.append(c); U.append(u); Z.append(z)
    return (np.array(O), np.array(C), np.array(U), np.array(Z))


# ---------------- model ----------------
class Encoder(nn.Module):
    """gru_priv: obs -> hp."""
    def __init__(self, obs_dim=8, fc_hid=64, out_dim=8):
        super().__init__()
        self.fc = nn.Linear(obs_dim, fc_hid)
        self.gru = nn.GRU(fc_hid, out_dim, batch_first=True)
    def forward(self, obs):
        x = torch.relu(self.fc(obs))
        h, _ = self.gru(x)
        return h


class Head(nn.Module):
    """head: [o, c_u, hp] -> u_hat. Optional z classifier on output."""
    def __init__(self, in_dim=20, out_dim=4, hid=64, with_z=False):
        super().__init__()
        self.with_z = with_z
        self.main = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim),
            nn.Sigmoid())
        if with_z:
            self.z_classifier = nn.Linear(out_dim, 1)
    def forward(self, x):
        aux = self.main(x)
        if self.with_z:
            return aux, self.z_classifier(aux)
        return aux


def train(obs, cu, u, with_z=False, lambda_z=1.0,
          n_steps=5000, batch_size=64, seed=0, lr=1e-3, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    obs_t = torch.tensor(obs, dtype=torch.float32)
    cu_t  = torch.tensor(cu,  dtype=torch.float32)
    u_t   = torch.tensor(u,   dtype=torch.float32)
    z_t   = (u_t[..., 0] > u_t[..., 1]).float()   # (N, T)

    N = obs_t.shape[0]
    encoder = Encoder()
    head = Head(with_z=with_z)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for step in range(n_steps):
        idx = np.random.choice(N, batch_size, replace=False)
        obs_b = obs_t[idx]; cu_b = cu_t[idx]; u_b = u_t[idx]; z_b = z_t[idx]

        h = encoder(obs_b)                          # (B, T, 8)
        x = torch.cat([obs_b, cu_b, h], dim=-1)     # (B, T, 20)

        if with_z:
            aux, z_logit = head(x)                  # (B, T, 4), (B, T, 1)
            loss_mse = F.mse_loss(aux, u_b)
            loss_z   = F.binary_cross_entropy_with_logits(z_logit.squeeze(-1), z_b)
            loss = loss_mse + lambda_z * loss_z
        else:
            aux = head(x)
            loss = F.mse_loss(aux, u_b)

        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 1000 == 0:
            print(f"  step {step:5d}  loss={loss.item():.5f}")

    return encoder, head


# ---------------- eval ----------------
def forward_all(encoder, head, obs, cu, shuffle_cu=False, zero_cu=False, seed=42):
    encoder.eval(); head.eval()
    cu_mod = cu.copy()
    if shuffle_cu:
        perm = np.random.RandomState(seed).permutation(len(cu))
        cu_mod = cu[perm]
    if zero_cu:
        cu_mod = np.zeros_like(cu)

    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32)
        cu_t  = torch.tensor(cu_mod, dtype=torch.float32)
        h = encoder(obs_t)
        x = torch.cat([obs_t, cu_t, h], dim=-1)
        out = head(x)
        if isinstance(out, tuple):
            aux, _ = out
        else:
            aux = out
    return aux.numpy()


def bacc_from_aux(aux, z, t):
    """bacc(sign(aux[t][0] - aux[t][1]) vs z)."""
    sign_pred = (aux[:, t, 0] > aux[:, t, 1]).astype(int)
    correct = float((sign_pred == z).mean())
    baccs = []
    for c in [0, 1]:
        m = z == c
        if m.sum() > 0:
            baccs.append(float((sign_pred[m] == c).mean()))
    return correct, float(np.mean(baccs))


# ---------------- main ----------------
def main():
    print("=" * 78)
    print("STEP B' — head_U training")
    print(f"ρ_u={RHO_U_NOMINAL}, σ_c={SIGMA_C_NOMINAL}, "
          f"σ_u={SIGMA_U}, OFFSET_U={OFFSET_U}")
    print("=" * 78)

    print(f"\ncollecting {N_TRAIN + N_VAL} episodes...")
    O, C, U, Z = collect_data(N_TRAIN + N_VAL, RHO_U_NOMINAL, SIGMA_C_NOMINAL)

    O_tr, C_tr, U_tr, Z_tr = O[:N_TRAIN], C[:N_TRAIN], U[:N_TRAIN], Z[:N_TRAIN]
    O_va, C_va, U_va, Z_va = O[N_TRAIN:], C[N_TRAIN:], U[N_TRAIN:], Z[N_TRAIN:]

    print(f"train: {len(Z_tr)}  P(z=1)={Z_tr.mean():.4f}")
    print(f"val:   {len(Z_va)}  P(z=1)={Z_va.mean():.4f}")

    results = {}

    for variant, with_z, lambda_z in [("V2_MSE_only", False, 0.0),
                                      ("V1_MSE_plus_BCE", True, 1.0)]:
        print(f"\n{'=' * 40}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 40}")

        encoder, head = train(O_tr, C_tr, U_tr, with_z=with_z,
                               lambda_z=lambda_z, n_steps=5000,
                               batch_size=64, seed=0, lr=1e-3, verbose=True)

        aux_normal  = forward_all(encoder, head, O_va, C_va)
        aux_shuffle = forward_all(encoder, head, O_va, C_va, shuffle_cu=True)
        aux_zero    = forward_all(encoder, head, O_va, C_va, zero_cu=True)

        acc_n, bacc_n = bacc_from_aux(aux_normal,  Z_va, T_LOCK)
        acc_s, bacc_s = bacc_from_aux(aux_shuffle, Z_va, T_LOCK)
        acc_z, bacc_z = bacc_from_aux(aux_zero,    Z_va, T_LOCK)

        # also evaluate at t=T_LOCK-1 for diagnostics
        acc_n4, bacc_n4 = bacc_from_aux(aux_normal, Z_va, T_LOCK - 1)

        # MSE on aux -> u on val
        u_hat = aux_normal
        mse_val = float(np.mean((u_hat - U_va) ** 2))

        print(f"\n  EVAL  aux(t=5) → z         acc={acc_n:.4f}  bacc={bacc_n:.4f}")
        print(f"  SHUF  aux(t=5) → z         acc={acc_s:.4f}  bacc={bacc_s:.4f}")
        print(f"  ZERO  aux(t=5) → z         acc={acc_z:.4f}  bacc={bacc_z:.4f}")
        print(f"  EVAL  aux(t=4) → z         acc={acc_n4:.4f}  bacc={bacc_n4:.4f}")
        print(f"  MSE(aux, u) on val        = {mse_val:.5f}")

        results[variant] = {
            "aux_z_t5_bacc":  bacc_n,
            "shuf_z_t5_bacc": bacc_s,
            "zero_z_t5_bacc": bacc_z,
            "aux_z_t4_bacc":  bacc_n4,
            "mse_val":        mse_val,
        }

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for variant, r in results.items():
        print(f"\n{variant}:")
        print(f"  aux(t=5) → z:     {r['aux_z_t5_bacc']:.4f}  "
              f"({'>=' if r['aux_z_t5_bacc'] >= 0.90 else '<'} 0.90)")
        print(f"  shuffle control:  {r['shuf_z_t5_bacc']:.4f}  "
              f"({'<=' if r['shuf_z_t5_bacc'] <= 0.60 else '>'} 0.60)")
        print(f"  zero control:     {r['zero_z_t5_bacc']:.4f}  "
              f"({'<=' if r['zero_z_t5_bacc'] <= 0.60 else '>'} 0.60)")

    print("\n" + "=" * 78)
    print("GATE DECISION")
    print("=" * 78)
    v2 = results["V2_MSE_only"]
    v2_pass = (v2["aux_z_t5_bacc"] >= 0.90 and
               v2["shuf_z_t5_bacc"] <= 0.60 and
               v2["zero_z_t5_bacc"] <= 0.60)
    print(f"  Variant 2 (MSE only, H1 test): "
          f"{'PASS' if v2_pass else 'FAIL'}")
    if v2_pass:
        print("  → Step B' passed. aux carries z, c_u is essential.")
        print("  → PPO may now be considered.")
    else:
        print("  → Step B' failed on Variant 2.")
        print("  → Check Variant 1 result above. If V1 passes and V2 fails,")
        print("    MSE supervision does not force z-preservation.")
        print("    Design decision required, no PPO.")

    with open("step_b_prime_v4_4b_result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to step_b_prime_v4_4b_result.json")


if __name__ == "__main__":
    main()