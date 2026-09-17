# env_validation_v4_3_oos.py
# Out-of-sample validation. Те же критерии. Другие seeds.
# Никаких правок среды, порогов, архитектуры.

import numpy as np
import torch
import torch.nn as nn
import json, math
from dataclasses import dataclass
from typing import List

# ------------------- FROZEN ENV PARAMS (identical to v4.2) -------------------
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
PRED_DIM = 4
HIST_K = 8
HIST_DIM = PRED_DIM * HIST_K + 5 * HIST_K

# ------------------- OOS SEEDS (changed from v4.2) -------------------
ENV_SEEDS_VALIDATION = [6, 7, 8, 9, 10]     # v4.2 used [1..5]
DATA_SEED = 999                             # v4.2 used 12345
FOLD_SEED_BASE = 5000                       # v4.2 used 1000
N_FOLDS = 5

N_TRAIN_CFG = 7
N_TEST_CFG = 5

E1_LOWER, E1_UPPER = 0.25, 0.75
E4_LOWER, E4_UPPER = 0.25, 0.90
E4_HIST_RATIO = 0.75
E3_MATCH_TOL = 0.30
E2_REL_TOL = 0.10


@dataclass(frozen=True)
class TaskCfg:
    task_type: str
    distractor: int
    def key(self):
        return f"{self.task_type}_d{self.distractor}"


def make_all_tasks():
    return [TaskCfg(t, d) for t in ["reach","find","avoid","wait"] for d in [0,1,2]]


def split_tasks():
    all_tasks = make_all_tasks()
    train_keys = {"reach_d0","reach_d1","find_d0","find_d2",
                  "avoid_d0","wait_d0","wait_d1"}
    train = [t for t in all_tasks if t.key() in train_keys]
    test  = [t for t in all_tasks if t.key() not in train_keys]
    return train, test


class Environment:
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def reset(self, cfg, seed):
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg
        self.t = 0; self.done = False; self.prev_action = 4
        self.q = self.rng.uniform(0.25, 0.75, 4)
        self.s = self.rng.uniform(0.25, 0.75, 4)
        self.w = self.rng.uniform(0.25, 0.75, 4)
        self.v = self.rng.uniform(0.25, 0.75, 4)
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

    def _c_from_action(self, x, a):
        return float(np.clip(np.dot(M_MAT[:, a], x) / 4.0, 0.0, 1.0))

    def _p_exec(self, a):
        c_s = self._c_from_action(self.s, a)
        c_w = self._c_from_action(self.w, a)
        z = ALPHA*(c_s - THETA) + ALPHA*(c_w - THETA)
        return float(1.0/(1.0 + np.exp(-z)))

    def _effective_move(self, a):
        x, y = self.pos
        if a == 0:   new = (x, max(0, y-1))
        elif a == 1: new = (x, min(GRID-1, y+1))
        elif a == 2: new = (max(0, x-1), y)
        elif a == 3: new = (min(GRID-1, x+1), y)
        else:        return True, self.pos
        blocked = (new == self.pos) or (new in self.objects)
        return (not blocked), new

    def step(self, a):
        onehot = np.zeros(5); onehot[a] = 1.0
        q_t = self.q.copy()
        eps_s = self.rng.normal(0, SIGMA_X, 4)
        eps_w = self.rng.normal(0, SIGMA_X, 4)
        eps_v = self.rng.normal(0, SIGMA_X, 4)
        self.s = np.clip(D_VEC + A_SCALAR*self.s + B_MAT@onehot + C_SCALAR*q_t + eps_s, 0, 1)
        self.w = np.clip(D_VEC + A_SCALAR*self.w + B_MAT@onehot + C_SCALAR*q_t + eps_w, 0, 1)
        self.v = np.clip(D_VEC + A_SCALAR*self.v + B_MAT@onehot + C_SCALAR*q_t + eps_v, 0, 1)
        p = self._p_exec(a)
        exec_ok = bool(self.rng.rand() < p)
        if exec_ok:
            move_ok, new_pos = self._effective_move(a)
            if move_ok: self.pos = new_pos; success = 1
            else: success = 0
        else: success = 0
        eta_q = self.rng.normal(0, SIGMA_Q, 4)
        self.q = np.clip(RHO_Q*q_t + OFFSET_Q + eta_q, 0, 1)
        self.signal_timer -= 1
        if self.signal_timer <= 0: self.signal_present = True
        self.t += 1
        reward = -0.01
        if self.cfg.task_type in ("reach","find") and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.cfg.task_type == "avoid" and self.pos in self.objects:
            reward = -1.0; self.done = True
        if self.cfg.task_type == "wait" and self.signal_present and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.t >= T_HORIZON: self.done = True
        self.prev_action = a
        self._compute_obs()
        return self.obs, reward, self.done, success

    def _compute_obs(self):
        nu = self.rng.normal(0, SIGMA_OBS, 4)
        y_q = np.clip(self.q + nu, 0, 1)
        x, yy = self.pos; gx, gy = self.goal
        dx = (gx - x)/GRID; dy = (gy - yy)/GRID
        if self.objects:
            dmin = min(abs(ox-x)+abs(oy-yy) for ox,oy in self.objects)/(2*GRID)
        else: dmin = 1.0
        sig = 1.0 if self.signal_present else 0.0
        self.obs = np.concatenate([y_q, [dx, dy, dmin, sig]])


def flatten_history_pred(obs_h, act_h):
    while len(obs_h) < HIST_K:
        obs_h = [np.zeros(PRED_DIM)] + obs_h
        act_h = [4] + act_h
    obs_h = obs_h[-HIST_K:]; act_h = act_h[-HIST_K:]
    o = np.concatenate(obs_h)
    a_oh = []
    for a in act_h:
        v = np.zeros(5); v[a] = 1.0; a_oh.append(v)
    return np.concatenate([o, np.concatenate(a_oh)]).astype(np.float32)


def collect_data(env, tasks, seeds, n_ep=5, data_seed=DATA_SEED):
    cur_in, hist_in = [], []
    s_t, w_t, v_t = [], [], []
    s_next, w_next, v_next = [], [], []
    rng = np.random.RandomState(data_seed)
    for cfg in tasks:
        for sd in seeds:
            for _ in range(n_ep):
                o = env.reset(cfg, sd)
                done = False; hist_o, hist_a = [], []
                while not done:
                    a = int(rng.randint(0, 5))
                    o_pred = o[:PRED_DIM].copy()
                    hist_input = flatten_history_pred(hist_o + [o_pred], hist_a + [a])
                    cur_in.append(o_pred); hist_in.append(hist_input)
                    s_t.append(env.s.copy()); w_t.append(env.w.copy()); v_t.append(env.v.copy())
                    o_next, r, done, succ = env.step(a)
                    s_next.append(env.s.copy()); w_next.append(env.w.copy()); v_next.append(env.v.copy())
                    hist_o.append(o_pred); hist_a.append(a)
                    if len(hist_o) > HIST_K:
                        hist_o = hist_o[-HIST_K:]; hist_a = hist_a[-HIST_K:]
                    o = o_next
    return {"o": np.array(cur_in), "hist": np.array(hist_in),
            "s": np.array(s_t), "w": np.array(w_t), "v": np.array(v_t),
            "s_next": np.array(s_next), "w_next": np.array(w_next),
            "v_next": np.array(v_next)}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)


def train_predictor(X_tr, Y_tr, X_va, Y_va, seed, in_dim, n_steps=3000, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X_tr, dtype=torch.float32); Yt = torch.tensor(Y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32); Yv = torch.tensor(Y_va, dtype=torch.float32)
    best_val = float("inf"); best_state = None; patience = 500; since_improve = 0
    for step in range(n_steps):
        model.train()
        idx = torch.randperm(Xt.shape[0])[:256]
        loss = ((model(Xt[idx]) - Yt[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = ((model(Xv) - Yv)**2).mean().item()
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                since_improve = 0
            else:
                since_improve += 100
            if since_improve >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model


def normalized_mse(pred, target):
    mse = float(np.mean((pred - target)**2))
    var = float(np.mean((target - target.mean(axis=0))**2))
    return mse / max(var, 1e-12)


def marginal_nll_per_dim(x):
    mu = np.mean(x, axis=0); var = np.var(x, axis=0) + 1e-6
    return 0.5*np.log(2*np.pi*var) + 0.5*(x-mu)**2/var


def gaussian_nll_from_mse(mse_dim):
    return 0.5*np.log(2*np.pi*mse_dim + 1e-12) + 0.5


def compute_metrics(data, target):
    Y = data[f"{target}_next"]; hist = data["hist"]; cur = data["o"]
    n = len(Y)
    Gs, U_norms = [], []
    mse_h_folds, mse_c_folds, var_folds = [], [], []
    for fold in range(N_FOLDS):
        idx = np.random.RandomState(FOLD_SEED_BASE + fold).permutation(n)
        tr = idx[: n*3//4]; va = idx[n*3//4:]
        hm = train_predictor(hist[tr], Y[tr], hist[va], Y[va],
                             seed=6000 + fold, in_dim=HIST_DIM)
        cm = train_predictor(cur[tr], Y[tr], cur[va], Y[va],
                             seed=7000 + fold, in_dim=PRED_DIM)
        with torch.no_grad():
            hp = hm(torch.tensor(hist[va], dtype=torch.float32)).numpy()
            cp = cm(torch.tensor(cur[va], dtype=torch.float32)).numpy()
        mse_h = np.mean((hp - Y[va])**2, axis=0)
        mse_c = np.mean((cp - Y[va])**2, axis=0)
        var = np.var(Y[va], axis=0)
        Gs.append(float(np.sum(gaussian_nll_from_mse(mse_h))) * -1 +
                  float(np.mean(np.sum(marginal_nll_per_dim(Y[va]), axis=1))))
        U_norms.append(normalized_mse(hp, Y[va]))
        mse_h_folds.append(mse_h); mse_c_folds.append(mse_c); var_folds.append(var)
    mse_h = np.mean(mse_h_folds, axis=0)
    mse_c = np.mean(mse_c_folds, axis=0)
    var = np.mean(var_folds, axis=0)
    ratio = mse_h / (mse_c + 1e-12)
    corridor = bool(np.all((E4_LOWER*var) < mse_c) and np.all(mse_c < (E4_UPPER*var)))
    hist_ok = bool(np.all(ratio <= E4_HIST_RATIO))
    return {
        "U_norm": float(np.mean(U_norms)),
        "G_mean": float(np.mean(Gs)),
        "G_std": float(np.std(Gs)),
        "ratio_dim": ratio.tolist(),
        "mse_cur_over_var": (mse_c/var).tolist(),
        "corridor_ok": corridor, "hist_ok": hist_ok,
    }


def compute_e3(env_proto, tasks, seeds, target, x_L, x_H, n_per=30):
    per_a = []
    for a in range(5):
        pL_list, pH_list = [], []
        for cfg in tasks:
            for sd in seeds:
                for ep in range(n_per):
                    seed_ep = sd*100000 + (hash(cfg.key())%10000)*100 + ep + 7777
                    eL = Environment(seed_ep); eL.reset(cfg, seed_ep)
                    if target == "s": eL.s = x_L.copy()
                    elif target == "w": eL.w = x_L.copy()
                    elif target == "v": eL.v = x_L.copy()
                    pL_list.append(eL._p_exec(a))
                    eH = Environment(seed_ep); eH.reset(cfg, seed_ep)
                    if target == "s": eH.s = x_H.copy()
                    elif target == "w": eH.w = x_H.copy()
                    elif target == "v": eH.v = x_H.copy()
                    pH_list.append(eH._p_exec(a))
        pL = float(np.mean(pL_list)); pH = float(np.mean(pH_list))
        eps = 1e-8
        pLc = min(max(pL, eps), 1-eps); pHc = min(max(pH, eps), 1-eps)
        kl = float(pLc*np.log(pLc/pHc) + (1-pLc)*np.log((1-pLc)/(1-pHc)))
        per_a.append(kl)
    return float(np.mean(per_a))


def main():
    print("=" * 60)
    print("ENV VALIDATION v4.3 — OUT-OF-SAMPLE")
    print("=" * 60)
    print(f"env seeds: {ENV_SEEDS_VALIDATION}, data seed: {DATA_SEED}, "
          f"fold base: {FOLD_SEED_BASE}")
    print()

    train_tasks, _ = split_tasks()
    env = Environment(seed=42)
    data = collect_data(env, train_tasks, ENV_SEEDS_VALIDATION, n_ep=5)
    print(f"data samples: {len(data['o'])}")

    metrics = {}
    for t in ["s", "w", "v"]:
        m = compute_metrics(data, t)
        metrics[t] = m
        print(f"\n--- {t} ---")
        print(f"  U_norm      = {m['U_norm']:.4f}")
        print(f"  G mean±std  = {m['G_mean']:.4f} ± {m['G_std']:.4f}")
        print(f"  ratio_dim   = {np.array2string(np.array(m['ratio_dim']), precision=4)}")
        print(f"  cur/var     = {np.array2string(np.array(m['mse_cur_over_var']), precision=4)}")
        print(f"  corridor={m['corridor_ok']} hist={m['hist_ok']}")

    print("\n--- E3 ---")
    x_L = np.array([0.2]*4); x_H = np.array([0.8]*4)
    E_s = compute_e3(env, train_tasks[:2], ENV_SEEDS_VALIDATION[:2], "s", x_L, x_H)
    E_w = compute_e3(env, train_tasks[:2], ENV_SEEDS_VALIDATION[:2], "w", x_L, x_H)
    E_v = compute_e3(env, train_tasks[:2], ENV_SEEDS_VALIDATION[:2], "v", x_L, x_H)
    e3r = abs(E_s-E_w)/max(E_s, E_w, 1e-12)
    print(f"  E_s={E_s:.6f} E_w={E_w:.6f} E_v={E_v:.6f} match={e3r:.3f}")

    print("\n--- E2 (10% primary) ---")
    e2_pass = True
    for i, j in [("s","w"),("s","v"),("w","v")]:
        gi, gj = metrics[i]["G_mean"], metrics[j]["G_mean"]
        rel = abs(gi-gj)/(0.5*(gi+gj)) if gi>0 and gj>0 else float("inf")
        ok = rel <= E2_REL_TOL
        print(f"  {i}/{j}: G={gi:.4f}/{gj:.4f} rel={rel:.4f} pass={ok}")
        if not ok: e2_pass = False

    e1_pass = all(E1_LOWER <= metrics[t]["U_norm"] <= E1_UPPER for t in ["s","w","v"])
    e3_pass = (E_s > 0 and E_w > 0 and E_v < 0.05 and e3r <= E3_MATCH_TOL)
    e4_pass = all(metrics[t]["corridor_ok"] and metrics[t]["hist_ok"] for t in ["s","w"])

    print("\n" + "=" * 60)
    print(f"  E1: {e1_pass}")
    print(f"  E2: {e2_pass}")
    print(f"  E3: {e3_pass}")
    print(f"  E4: {e4_pass}")
    overall = e1_pass and e2_pass and e3_pass and e4_pass
    print(f"  OVERALL (OOS): {'PASSED' if overall else 'FAILED'}")
    print("=" * 60)

    with open("env_validation_v4_3_oos_result.json", "w") as f:
        json.dump({"metrics": {k: v for k, v in metrics.items()},
                   "E3": {"E_s": E_s, "E_w": E_w, "E_v": E_v, "match_ratio": e3r},
                   "e1_pass": e1_pass, "e2_pass": e2_pass,
                   "e3_pass": e3_pass, "e4_pass": e4_pass,
                   "overall_pass": overall}, f, indent=2, default=str)


if __name__ == "__main__":
    main()