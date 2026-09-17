# env_validation_v4_2.py
# E2 = 10% relative (primary), CI = diagnostic.
# Prediction predictor sees only q-proxy (4 dims), not task features.
# This removes downstream contamination of E4.

import numpy as np
import torch
import torch.nn as nn
import json, math
from dataclasses import dataclass
from typing import List

A_SCALAR = 0.20
B_MAT = np.array([
    [-0.025, -0.010,  0.000,  0.015,  0.005],
    [-0.0075, -0.0225, 0.010,  0.000,  0.0125],
    [ 0.000, -0.0125, -0.020,  0.010,  0.005],
    [-0.010,  0.000, -0.0125, 0.0075, -0.0175],
], dtype=np.float64)
C_SCALAR = 0.40
D_VEC = np.array([0.20]*4, dtype=np.float64)
SIGMA_X = 0.035
RHO_Q = 0.98
OFFSET_Q = 0.01
SIGMA_Q = 0.020
VAR_Q_POP = SIGMA_Q**2 / (1 - RHO_Q**2)
SIGMA_OBS = 0.200

M_MAT = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
], dtype=np.float64)
THETA = 0.5
ALPHA = 8.0
GRID = 8
T_HORIZON = 100
OBS_DIM = 8          # full env obs (for future agent)
PRED_DIM = 4         # q-proxy only (for predictors)
HIST_K = 8
HIST_DIM = PRED_DIM * HIST_K + 5 * HIST_K  # 4*8 + 5*8 = 72

ENV_SEEDS = list(range(1, 11))
N_TRAIN_CFG = 7
N_TEST_CFG = 5

E1_LOWER, E1_UPPER = 0.25, 0.75
E4_LOWER, E4_UPPER = 0.25, 0.90
E4_HIST_RATIO = 0.75
E3_MATCH_TOL = 0.30
E2_REL_TOL = 0.10
N_FOLDS = 5


@dataclass(frozen=True)
class TaskCfg:
    task_type: str
    distractor: int
    def key(self):
        return f"{self.task_type}_d{self.distractor}"


def make_all_tasks():
    tasks = []
    for t in ["reach", "find", "avoid", "wait"]:
        for d in [0, 1, 2]:
            tasks.append(TaskCfg(t, d))
    return tasks


def split_tasks():
    all_tasks = make_all_tasks()
    train_keys = {"reach_d0","reach_d1","find_d0","find_d2",
                  "avoid_d0","wait_d0","wait_d1"}
    train = [t for t in all_tasks if t.key() in train_keys]
    test  = [t for t in all_tasks if t.key() not in train_keys]
    assert len(train) == N_TRAIN_CFG
    assert len(test) == N_TEST_CFG
    return train, test


class Environment:
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def reset(self, cfg, seed):
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg
        self.t = 0
        self.done = False
        self.prev_action = 4

        self.q = self.rng.uniform(0.25, 0.75, 4).astype(np.float64)
        self.s = self.rng.uniform(0.25, 0.75, 4).astype(np.float64)
        self.w = self.rng.uniform(0.25, 0.75, 4).astype(np.float64)
        self.v = self.rng.uniform(0.25, 0.75, 4).astype(np.float64)

        self.pos = (int(self.rng.randint(0, GRID)), int(self.rng.randint(0, GRID)))
        while True:
            gx = int(self.rng.randint(0, GRID)); gy = int(self.rng.randint(0, GRID))
            if (gx, gy) != self.pos: break
        self.goal = (gx, gy)

        self.objects = set()
        for _ in range(cfg.distractor):
            while True:
                ox = int(self.rng.randint(0, GRID)); oy = int(self.rng.randint(0, GRID))
                if (ox, oy) != self.pos and (ox, oy) != self.goal: break
            self.objects.add((ox, oy))

        self.signal_present = False
        self.signal_timer = int(self.rng.randint(20, 60))
        self._compute_obs()
        return self.obs

    def _c_from_action(self, x, action):
        return float(np.clip(np.dot(M_MAT[:, action], x) / 4.0, 0.0, 1.0))

    def _p_exec(self, action):
        c_s = self._c_from_action(self.s, action)
        c_w = self._c_from_action(self.w, action)
        z = ALPHA * (c_s - THETA) + ALPHA * (c_w - THETA)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _effective_move(self, action):
        x, y = self.pos
        if action == 0:   new = (x, max(0, y - 1))
        elif action == 1: new = (x, min(GRID - 1, y + 1))
        elif action == 2: new = (max(0, x - 1), y)
        elif action == 3: new = (min(GRID - 1, x + 1), y)
        else:             return True, self.pos
        blocked = (new == self.pos) or (new in self.objects)
        return (not blocked), new

    def step(self, action):
        onehot = np.zeros(5); onehot[action] = 1.0
        q_t = self.q.copy()
        eps_s = self.rng.normal(0, SIGMA_X, 4)
        eps_w = self.rng.normal(0, SIGMA_X, 4)
        eps_v = self.rng.normal(0, SIGMA_X, 4)

        self.s = np.clip(D_VEC + A_SCALAR*self.s + B_MAT @ onehot + C_SCALAR*q_t + eps_s, 0, 1)
        self.w = np.clip(D_VEC + A_SCALAR*self.w + B_MAT @ onehot + C_SCALAR*q_t + eps_w, 0, 1)
        self.v = np.clip(D_VEC + A_SCALAR*self.v + B_MAT @ onehot + C_SCALAR*q_t + eps_v, 0, 1)

        p = self._p_exec(action)
        exec_ok = bool(self.rng.rand() < p)

        if exec_ok:
            move_ok, new_pos = self._effective_move(action)
            if move_ok:
                self.pos = new_pos; success = 1
            else:
                success = 0
        else:
            success = 0

        eta_q = self.rng.normal(0, SIGMA_Q, 4)
        self.q = np.clip(RHO_Q * q_t + OFFSET_Q + eta_q, 0, 1)

        self.signal_timer -= 1
        if self.signal_timer <= 0:
            self.signal_present = True

        self.t += 1
        reward = -0.01
        if self.cfg.task_type in ("reach", "find") and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.cfg.task_type == "avoid" and self.pos in self.objects:
            reward = -1.0; self.done = True
        if self.cfg.task_type == "wait" and self.signal_present and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.t >= T_HORIZON:
            self.done = True

        self.prev_action = action
        self._compute_obs()
        return self.obs, reward, self.done, success

    def _compute_obs(self):
        nu = self.rng.normal(0, SIGMA_OBS, 4)
        y_q = np.clip(self.q + nu, 0, 1)
        x, yy = self.pos
        gx, gy = self.goal
        dx = (gx - x) / GRID
        dy = (gy - yy) / GRID
        if self.objects:
            dmin = min(abs(ox - x) + abs(oy - yy)
                       for ox, oy in self.objects) / (2 * GRID)
        else:
            dmin = 1.0
        sig = 1.0 if self.signal_present else 0.0
        self.obs = np.concatenate([y_q, [dx, dy, dmin, sig]]).astype(np.float64)


def flatten_history_pred(obs_pred_hist, act_hist):
    """Pad with zeros to HIST_K. Only uses PRED_DIM dims per obs."""
    while len(obs_pred_hist) < HIST_K:
        obs_pred_hist = [np.zeros(PRED_DIM)] + obs_pred_hist
        act_hist = [4] + act_hist
    obs_pred_hist = obs_pred_hist[-HIST_K:]
    act_hist = act_hist[-HIST_K:]
    o = np.concatenate(obs_pred_hist)
    a_oh = []
    for a in act_hist:
        v = np.zeros(5); v[a] = 1.0
        a_oh.append(v)
    a = np.concatenate(a_oh)
    return np.concatenate([o, a]).astype(np.float32)


def collect_data(env, tasks, seeds, n_episodes_per_cfg=5):
    cur_in, hist_in = [], []
    s_t, w_t, v_t = [], [], []
    s_next, w_next, v_next = [], [], []

    rng = np.random.RandomState(12345)

    for cfg in tasks:
        for sd in seeds:
            for _ in range(n_episodes_per_cfg):
                o = env.reset(cfg, sd)
                done = False
                hist_o, hist_a = [], []
                while not done:
                    a = int(rng.randint(0, 5))
                    o_pred = o[:PRED_DIM].copy()
                    h_o = hist_o + [o_pred]
                    h_a = hist_a + [a]
                    hist_input = flatten_history_pred(h_o, h_a)

                    cur_in.append(o_pred)
                    hist_in.append(hist_input)

                    s_t.append(env.s.copy())
                    w_t.append(env.w.copy())
                    v_t.append(env.v.copy())

                    o_next, r, done, succ = env.step(a)

                    s_next.append(env.s.copy())
                    w_next.append(env.w.copy())
                    v_next.append(env.v.copy())

                    hist_o.append(o_pred); hist_a.append(a)
                    if len(hist_o) > HIST_K:
                        hist_o = hist_o[-HIST_K:]; hist_a = hist_a[-HIST_K:]
                    o = o_next

    return {
        "o": np.array(cur_in),       # shape (N, PRED_DIM)
        "hist": np.array(hist_in),   # shape (N, HIST_DIM)
        "s": np.array(s_t), "w": np.array(w_t), "v": np.array(v_t),
        "s_next": np.array(s_next), "w_next": np.array(w_next), "v_next": np.array(v_next),
    }


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_predictor(X_tr, Y_tr, X_va, Y_va, seed, in_dim, n_steps=3000, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    Yt = torch.tensor(Y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    Yv = torch.tensor(Y_va, dtype=torch.float32)
    best_val = float("inf"); best_state = None
    patience = 500; since_improve = 0
    for step in range(n_steps):
        model.train()
        idx = torch.randperm(Xt.shape[0])[:256]
        pred = model(Xt[idx])
        loss = ((pred - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = ((model(Xv) - Yv) ** 2).mean().item()
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                since_improve = 0
            else:
                since_improve += 100
            if since_improve >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def normalized_mse(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    var = float(np.mean((target - target.mean(axis=0)) ** 2))
    return mse / max(var, 1e-12)


def marginal_nll_per_dim(x):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0) + 1e-6
    diff = x - mu
    nll_dim = 0.5 * np.log(2 * np.pi * var) + 0.5 * diff ** 2 / var
    return nll_dim


def gaussian_nll_from_mse_per_dim(mse_dim):
    return 0.5 * np.log(2 * np.pi * mse_dim + 1e-12) + 0.5


def compute_metrics_for_target(data, target, n_folds=N_FOLDS):
    Y = data[f"{target}_next"]
    hist = data["hist"]
    cur = data["o"]
    n = len(Y)

    Gs = []; U_norms = []
    mse_hist_dim_folds = []; mse_cur_dim_folds = []; var_dim_folds = []

    for fold in range(n_folds):
        idx = np.random.RandomState(1000 + fold).permutation(n)
        tr = idx[: n * 3 // 4]; va = idx[n * 3 // 4:]

        hist_model = train_predictor(hist[tr], Y[tr], hist[va], Y[va],
                                     seed=2000 + fold, in_dim=HIST_DIM)
        cur_model = train_predictor(cur[tr], Y[tr], cur[va], Y[va],
                                    seed=3000 + fold, in_dim=PRED_DIM)
        with torch.no_grad():
            hist_pred = hist_model(torch.tensor(hist[va], dtype=torch.float32)).numpy()
            cur_pred = cur_model(torch.tensor(cur[va], dtype=torch.float32)).numpy()

        mse_hist_dim = np.mean((hist_pred - Y[va]) ** 2, axis=0)
        mse_cur_dim = np.mean((cur_pred - Y[va]) ** 2, axis=0)
        var_dim = np.var(Y[va], axis=0)

        U_norm_fold = normalized_mse(hist_pred, Y[va])
        nll_hist = float(np.sum(gaussian_nll_from_mse_per_dim(mse_hist_dim)))
        nll_marg_dim = marginal_nll_per_dim(Y[va])
        nll_marg = float(np.mean(np.sum(nll_marg_dim, axis=1)))
        G_fold = nll_marg - nll_hist

        Gs.append(G_fold); U_norms.append(U_norm_fold)
        mse_hist_dim_folds.append(mse_hist_dim)
        mse_cur_dim_folds.append(mse_cur_dim)
        var_dim_folds.append(var_dim)

    mse_hist_dim = np.mean(mse_hist_dim_folds, axis=0)
    mse_cur_dim = np.mean(mse_cur_dim_folds, axis=0)
    var_dim = np.mean(var_dim_folds, axis=0)
    ratio = mse_hist_dim / (mse_cur_dim + 1e-12)

    corridor_ok = bool(np.all((E4_LOWER * var_dim) < mse_cur_dim) and
                       np.all(mse_cur_dim < (E4_UPPER * var_dim)))
    hist_ok = bool(np.all(ratio <= E4_HIST_RATIO))

    leak_ok = True
    for j in range(4):
        src = cur[:, min(j, PRED_DIM - 1)]
        if np.std(src) > 1e-9 and np.std(Y[:, j]) > 1e-9:
            r = np.corrcoef(src, Y[:, j])[0, 1]
            if abs(r) > 0.5: leak_ok = False

    return {
        "U_norm": float(np.mean(U_norms)),
        "U_norm_folds": U_norms,
        "G_mean": float(np.mean(Gs)),
        "G_std": float(np.std(Gs)),
        "G_all": Gs,
        "mse_hist_dim": mse_hist_dim.tolist(),
        "mse_cur_dim": mse_cur_dim.tolist(),
        "var_dim": var_dim.tolist(),
        "ratio_dim": ratio.tolist(),
        "mse_cur_over_var": (mse_cur_dim / var_dim).tolist(),
        "corridor_ok": corridor_ok,
        "hist_ok": hist_ok,
        "leak_ok": leak_ok,
    }


def compute_e3(env_proto, tasks, seeds, target, x_L, x_H, n_per=30):
    per_action = []
    for a in range(5):
        pL_list, pH_list = [], []
        for cfg in tasks:
            for sd in seeds:
                for ep in range(n_per):
                    seed_ep = sd * 100000 + (hash(cfg.key()) % 10000)*100 + ep
                    env_L = Environment(seed_ep); env_L.reset(cfg, seed_ep)
                    if target == "s": env_L.s = x_L.copy()
                    elif target == "w": env_L.w = x_L.copy()
                    elif target == "v": env_L.v = x_L.copy()
                    pL_list.append(env_L._p_exec(a))
                    env_H = Environment(seed_ep); env_H.reset(cfg, seed_ep)
                    if target == "s": env_H.s = x_H.copy()
                    elif target == "w": env_H.w = x_H.copy()
                    elif target == "v": env_H.v = x_H.copy()
                    pH_list.append(env_H._p_exec(a))
        pL = float(np.mean(pL_list)); pH = float(np.mean(pH_list))
        eps = 1e-8
        pLc = min(max(pL, eps), 1-eps); pHc = min(max(pH, eps), 1-eps)
        kl = float(pLc*np.log(pLc/pHc) + (1-pLc)*np.log((1-pLc)/(1-pHc)))
        per_action.append({"action": a, "pL": pL, "pH": pH, "abs_delta": abs(pL-pH), "kl": kl})
    E = float(np.mean([r["kl"] for r in per_action]))
    return E, per_action


def sanity_checks():
    rng = np.random.RandomState(0)
    y = rng.uniform(0, 1, (500, 4))
    U_oracle = normalized_mse(y.copy(), y)
    mean_pred = np.repeat(y.mean(axis=0, keepdims=True), len(y), axis=0)
    U_constant = normalized_mse(mean_pred, y)
    print("Sanity:")
    print(f"  U_oracle   = {U_oracle:.2e}")
    print(f"  U_constant = {U_constant:.6f}")
    assert U_oracle < 1e-10
    assert 0.95 <= U_constant <= 1.05


def main():
    print("=" * 60)
    print("ENV VALIDATION v4.2 — E2=10% primary; prediction via q-proxy only")
    print("=" * 60)
    print(f"PRED_DIM={PRED_DIM}, HIST_DIM={HIST_DIM}")
    sanity_checks()
    print("Sanity OK\n")

    train_tasks, _ = split_tasks()
    env = Environment(seed=42)
    data = collect_data(env, train_tasks, ENV_SEEDS[:5], n_episodes_per_cfg=5)
    print(f"data samples: {len(data['o'])}")

    metrics = {}
    for target in ["s", "w", "v"]:
        m = compute_metrics_for_target(data, target)
        metrics[target] = m
        print(f"\n--- target={target} ---")
        print(f"  U_norm       = {m['U_norm']:.4f}")
        print(f"  G mean ± std = {m['G_mean']:.4f} ± {m['G_std']:.4f}")
        print(f"  ratio_dim    = {np.array2string(np.array(m['ratio_dim']), precision=4)}")
        print(f"  mse_cur/var  = {np.array2string(np.array(m['mse_cur_over_var']), precision=4)}")
        print(f"  corridor_ok  = {m['corridor_ok']}, hist_ok = {m['hist_ok']}")

    print("\n--- E3 ---")
    x_L = np.array([0.2]*4); x_H = np.array([0.8]*4)
    E_s, _ = compute_e3(env, train_tasks[:2], ENV_SEEDS[:2], "s", x_L, x_H)
    E_w, _ = compute_e3(env, train_tasks[:2], ENV_SEEDS[:2], "w", x_L, x_H)
    E_v, _ = compute_e3(env, train_tasks[:2], ENV_SEEDS[:2], "v", x_L, x_H)
    e3_ratio = abs(E_s - E_w) / max(E_s, E_w, 1e-12)
    print(f"  E_s={E_s:.6f} E_w={E_w:.6f} E_v={E_v:.6f} match_ratio={e3_ratio:.3f}")

    print("\n--- E2 (primary: relative ≤ 10%) ---")
    e2_pass = True
    for (i, j) in [("s","w"), ("s","v"), ("w","v")]:
        gi = metrics[i]["G_mean"]; gj = metrics[j]["G_mean"]
        if gi <= 0 or gj <= 0:
            print(f"  {i}/{j}: non-positive G, FAIL")
            e2_pass = False; continue
        rel = abs(gi - gj) / (0.5 * (gi + gj))
        ok = rel <= E2_REL_TOL
        print(f"  {i}/{j}: G={gi:.4f}/{gj:.4f} rel={rel:.4f} pass={ok}")
        if not ok: e2_pass = False

    print("\n--- E2 diagnostic (95% CI of log ratio, 5 folds) ---")
    for (i, j) in [("s","w"), ("s","v"), ("w","v")]:
        gi = np.array(metrics[i]["G_all"]); gj = np.array(metrics[j]["G_all"])
        log_r = np.log(gi / gj)
        lo, hi = np.percentile(log_r, [2.5, 97.5])
        print(f"  {i}/{j}: CI=[{lo:.4f},{hi:.4f}] ∋0={(lo<=0<=hi)}")

    print("\n" + "=" * 60)
    print("CHECKS")
    print("=" * 60)
    e1_pass = all(E1_LOWER <= metrics[t]["U_norm"] <= E1_UPPER for t in ["s","w","v"])
    e3_pass = (E_s > 0 and E_w > 0 and E_v < 0.05 and e3_ratio <= E3_MATCH_TOL)
    e4_pass = all(metrics[t]["corridor_ok"] and metrics[t]["hist_ok"] for t in ["s","w"])
    leak_pass = metrics["s"]["leak_ok"] and metrics["w"]["leak_ok"]

    print(f"  E1:                         {e1_pass}")
    print(f"  E2 (rel ≤ {E2_REL_TOL}):           {e2_pass}")
    print(f"  E3:                         {e3_pass}")
    print(f"  E4 (corridor + hist≤{E4_HIST_RATIO}): {e4_pass}")
    print(f"  leak:                       {leak_pass}")
    overall = e1_pass and e2_pass and e3_pass and e4_pass and leak_pass
    print("\n" + ("PASSED" if overall else "FAILED"))

    out = {
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "G_all"}
                    for k, v in metrics.items()},
        "E3": {"E_s": E_s, "E_w": E_w, "E_v": E_v, "match_ratio": e3_ratio},
        "e1_pass": e1_pass, "e2_pass": e2_pass, "e3_pass": e3_pass,
        "e4_pass": e4_pass, "leak_pass": leak_pass, "overall_pass": overall,
    }
    with open("env_validation_v4_2_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()