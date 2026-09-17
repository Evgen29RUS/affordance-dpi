# classify_probe_v4_3.py
# Probes on v4.3 substrate. Environment unchanged. PPO not invoked.
#
# Measures:
#   hist_pred (72-dim: q-proxy x 8 + action one-hot x 8) -> b_s, b_w, b_xor
#   hist_full (104-dim: full 8-d obs x 8 + action one-hot x 8) -> b_s, b_w, b_xor
#   cur       (4-dim: current q-proxy) -> b_s, b_w, b_xor   [negative control]
#
# Targets are all on t+1:
#   b_s   = 1[ s_{t+1}[0] > s_{t+1}[1] ]
#   b_w   = 1[ w_{t+1}[0] > w_{t+1}[1] ]
#   b_xor = b_s XOR b_w
#
# Classification is preregistered in main().

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# ------------------- FROZEN ENV PARAMS (identical to v4.3) -------------------
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

THETA   = 0.5
ALPHA   = 8.0
GRID    = 8
T_HORIZON = 100

PRED_DIM  = 4
FULL_DIM  = 8
HIST_K    = 8
HIST_DIM_PRED = PRED_DIM * HIST_K + 5 * HIST_K   # 72
HIST_DIM_FULL = FULL_DIM * HIST_K + 5 * HIST_K   # 104

ENV_SEEDS = [6, 7, 8, 9, 10]
DATA_SEED = 999


# ------------------- TASKS -------------------
@dataclass(frozen=True)
class TaskCfg:
    task_type: str
    distractor: int
    def key(self):
        return f"{self.task_type}_d{self.distractor}"

def train_tasks():
    keys = {"reach_d0", "reach_d1", "find_d0", "find_d2",
            "avoid_d0", "wait_d0", "wait_d1"}
    return [TaskCfg(t, d) for t in ["reach", "find", "avoid", "wait"] for d in [0, 1, 2]
            if f"{t}_d{d}" in keys]


# ------------------- ENVIRONMENT -------------------
class Environment:
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def reset(self, cfg, seed):
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg
        self.t = 0
        self.done = False
        self.prev_action = 4
        self.q = self.rng.uniform(0.25, 0.75, 4)
        self.s = self.rng.uniform(0.25, 0.75, 4)
        self.w = self.rng.uniform(0.25, 0.75, 4)
        self.v = self.rng.uniform(0.25, 0.75, 4)
        self.pos = (int(self.rng.randint(0, GRID)), int(self.rng.randint(0, GRID)))
        while True:
            gx = int(self.rng.randint(0, GRID))
            gy = int(self.rng.randint(0, GRID))
            if (gx, gy) != self.pos:
                break
        self.goal = (gx, gy)
        self.objects = set()
        for _ in range(cfg.distractor):
            while True:
                ox = int(self.rng.randint(0, GRID))
                oy = int(self.rng.randint(0, GRID))
                if (ox, oy) != self.pos and (ox, oy) != self.goal:
                    break
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
        z = ALPHA * (c_s - THETA) + ALPHA * (c_w - THETA)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _effective_move(self, a):
        x, y = self.pos
        if a == 0:   new = (x, max(0, y - 1))
        elif a == 1: new = (x, min(GRID - 1, y + 1))
        elif a == 2: new = (max(0, x - 1), y)
        elif a == 3: new = (min(GRID - 1, x + 1), y)
        else:        return True, self.pos
        blocked = (new == self.pos) or (new in self.objects)
        return (not blocked), new

    def step(self, a):
        onehot = np.zeros(5); onehot[a] = 1.0
        q_t = self.q.copy()
        eps_s = self.rng.normal(0, SIGMA_X, 4)
        eps_w = self.rng.normal(0, SIGMA_X, 4)
        eps_v = self.rng.normal(0, SIGMA_X, 4)

        self.s = np.clip(D_VEC + A_SCALAR*self.s + B_MAT @ onehot + C_SCALAR*q_t + eps_s, 0, 1)
        self.w = np.clip(D_VEC + A_SCALAR*self.w + B_MAT @ onehot + C_SCALAR*q_t + eps_w, 0, 1)
        self.v = np.clip(D_VEC + A_SCALAR*self.v + B_MAT @ onehot + C_SCALAR*q_t + eps_v, 0, 1)

        p = self._p_exec(a)
        exec_ok = bool(self.rng.rand() < p)
        if exec_ok:
            move_ok, new_pos = self._effective_move(a)
            if move_ok:
                self.pos = new_pos

        eta_q = self.rng.normal(0, SIGMA_Q, 4)
        self.q = np.clip(RHO_Q * q_t + OFFSET_Q + eta_q, 0, 1)

        self.signal_timer -= 1
        if self.signal_timer <= 0:
            self.signal_present = True

        self.t += 1
        reward = -0.01
        self.done = False
        if self.cfg.task_type in ("reach", "find") and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.cfg.task_type == "avoid" and self.pos in self.objects:
            reward = -1.0; self.done = True
        if self.cfg.task_type == "wait" and self.signal_present and self.pos == self.goal:
            reward = 1.0; self.done = True
        if self.t >= T_HORIZON:
            self.done = True

        self._compute_obs()
        return self.obs, reward, self.done

    def _compute_obs(self):
        nu = self.rng.normal(0, SIGMA_OBS, 4)
        y_q = np.clip(self.q + nu, 0, 1)
        x, yy = self.pos
        gx, gy = self.goal
        dx = (gx - x) / GRID
        dy = (gy - yy) / GRID
        if self.objects:
            dmin = min(abs(ox - x) + abs(oy - yy) for ox, oy in self.objects) / (2 * GRID)
        else:
            dmin = 1.0
        sig = 1.0 if self.signal_present else 0.0
        self.obs = np.concatenate([y_q, [dx, dy, dmin, sig]]).astype(np.float64)


# ------------------- DATA COLLECTION -------------------
def flatten_hist(obs_h, act_h, dim):
    while len(obs_h) < HIST_K:
        obs_h = [np.zeros(dim, dtype=np.float32)] + obs_h
        act_h = [4] + act_h
    obs_h = obs_h[-HIST_K:]
    act_h = act_h[-HIST_K:]
    a_oh = []
    for a in act_h:
        v = np.zeros(5, dtype=np.float32); v[a] = 1.0
        a_oh.append(v)
    return np.concatenate([np.concatenate(obs_h),
                           np.concatenate(a_oh)]).astype(np.float32)


def collect_with_traj(tasks, seeds, n_ep=8, data_seed=DATA_SEED):
    """
    For each step t:
      - input hist built from o_{0..t} (with a_{0..t} one-hots)
      - input cur = o_t[:PRED_DIM]
      - target s_{t+1}, w_{t+1} (overwrites stored s_t after env.step)
      - tid constant within episode
    """
    env = Environment(seed=42)
    rng = np.random.RandomState(data_seed)

    H_pred, H_full, O, S, W, TID = [], [], [], [], [], []
    tid = 0

    for cfg in tasks:
        for sd in seeds:
            for _ in range(n_ep):
                o = env.reset(cfg, sd)
                done = False
                hist_o_pred, hist_o_full, hist_a = [], [], []

                while not done:
                    a = int(rng.randint(0, 5))

                    o_pred = o[:PRED_DIM].astype(np.float32).copy()
                    o_full = o.astype(np.float32).copy()

                    hp = flatten_hist(hist_o_pred + [o_pred], hist_a + [a], PRED_DIM)
                    hf = flatten_hist(hist_o_full + [o_full], hist_a + [a], FULL_DIM)

                    H_pred.append(hp)
                    H_full.append(hf)
                    O.append(o_pred)
                    S.append(env.s.copy())   # placeholder; overwritten to s_{t+1} below
                    W.append(env.w.copy())
                    TID.append(tid)

                    o, _r, done = env.step(a)

                    # overwrite with the post-step value = s_{t+1}, w_{t+1}
                    S[-1] = env.s.copy()
                    W[-1] = env.w.copy()

                    hist_o_pred.append(o_pred)
                    hist_o_full.append(o_full)
                    hist_a.append(a)
                    if len(hist_a) > HIST_K:
                        hist_o_pred = hist_o_pred[-HIST_K:]
                        hist_o_full = hist_o_full[-HIST_K:]
                        hist_a = hist_a[-HIST_K:]

                tid += 1

    return (np.array(H_pred, dtype=np.float32),
            np.array(H_full, dtype=np.float32),
            np.array(O,      dtype=np.float32),
            np.array(S,      dtype=np.float64),
            np.array(W,      dtype=np.float64),
            np.array(TID,    dtype=np.int64))


# ------------------- PROBES -------------------
class LinProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2)
    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, d, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 2),
        )
    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, Xva, yva, model, seed, n_steps=4000, lr=1e-3):
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = -1.0
    best_state = None

    for step in range(n_steps):
        model.train()
        idx = torch.randperm(Xt.shape[0])[:256]
        logits = model(Xt[idx])
        loss = nn.functional.cross_entropy(logits, yt[idx])
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
    yv_np = yva

    acc = float((pred == yv_np).mean())
    baccs = []
    for c in [0, 1]:
        m = yv_np == c
        if m.sum() > 0:
            baccs.append(float((pred[m] == c).mean()))
    bacc = float(np.mean(baccs)) if baccs else float("nan")
    return acc, bacc


def evaluate_probe(X, y, traj_id, model_ctor, n_folds=5, base_seed=0):
    unique_tid = np.unique(traj_id)
    rng = np.random.RandomState(base_seed)
    rng.shuffle(unique_tid)
    folds = np.array_split(unique_tid, n_folds)

    accs, baccs = [], []
    for k, va_tids in enumerate(folds):
        va_mask = np.isin(traj_id, va_tids)
        tr_mask = ~va_mask
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        model = model_ctor(X.shape[1])
        acc, bacc = train_probe(
            X[tr_mask], y[tr_mask], X[va_mask], y[va_mask],
            model, seed=base_seed * 1000 + k,
        )
        accs.append(acc); baccs.append(bacc)

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(baccs))


# ------------------- MAIN -------------------
def main():
    print("=" * 78)
    print("CLASSIFY PROBE on v4.3 substrate  (env unchanged, PPO not invoked)")
    print("=" * 78)

    tasks = train_tasks()
    H_pred, H_full, O, S, W, TID = collect_with_traj(tasks, ENV_SEEDS, n_ep=8)
    print(f"samples: {len(H_pred)}   trajectories: {len(np.unique(TID))}")
    print(f"H_pred dim: {H_pred.shape[1]}   H_full dim: {H_full.shape[1]}")

    # targets on t+1
    y_s   = (S[:, 0] > S[:, 1]).astype(np.int64)
    y_w   = (W[:, 0] > W[:, 1]).astype(np.int64)
    y_xor = np.logical_xor(y_s, y_w).astype(np.int64)

    # ---- joint frequencies (must be checked before interpreting accuracies) ----
    print("\n--- joint frequencies ---")
    print(f"  P(b_s=1)   = {y_s.mean():.4f}")
    print(f"  P(b_w=1)   = {y_w.mean():.4f}")
    print(f"  P(b_xor=1) = {y_xor.mean():.4f}")
    for a in [0, 1]:
        for b in [0, 1]:
            print(f"  P(b_s={a}, b_w={b}) = {((y_s == a) & (y_w == b)).mean():.4f}")

    # ---- probes: 3 inputs x 3 targets x 2 model classes ----
    print("\n--- probes (5-fold by trajectory; acc / balanced acc) ---")
    rows = []
    inputs = [("hist_pred", H_pred), ("hist_full", H_full), ("cur", O)]
    targets = [("b_s", y_s), ("b_w", y_w), ("b_xor", y_xor)]

    for name_in, X in inputs:
        for name_tgt, y in targets:
            a_lin, s_lin, ba_lin = evaluate_probe(X, y, TID, LinProbe, base_seed=1)
            a_mlp, s_mlp, ba_mlp = evaluate_probe(X, y, TID, MLPProbe, base_seed=2)
            print(f"  {name_in:10s} -> {name_tgt:5s} | "
                  f"lin acc={a_lin:.4f}±{s_lin:.4f} bacc={ba_lin:.4f} | "
                  f"mlp acc={a_mlp:.4f}±{s_mlp:.4f} bacc={ba_mlp:.4f}")
            rows.append((name_in, name_tgt, a_lin, ba_lin, a_mlp, ba_mlp))

    # ---- preregistered classification ----
    print("\n" + "=" * 78)
    print("PREREGISTERED CLASSIFICATION")
    print("=" * 78)

    def find(inp, tgt):
        for r in rows:
            if r[0] == inp and r[1] == tgt:
                return r
        raise KeyError((inp, tgt))

    hx_pred = find("hist_pred", "b_xor")[4]
    hx_full = find("hist_full", "b_xor")[4]
    cx      = find("cur",       "b_xor")[4]
    cs      = find("cur",       "b_s")[4]
    cw      = find("cur",       "b_w")[4]

    print(f"hist_pred -> XOR (MLP acc): {hx_pred:.4f}")
    print(f"hist_full -> XOR (MLP acc): {hx_full:.4f}   <-- number that matters for the agent")
    print(f"cur       -> XOR (MLP acc): {cx:.4f}   (sanity, expect <= 0.55)")
    print(f"cur       -> b_s (MLP acc): {cs:.4f}   (sanity, expect <= 0.58)")
    print(f"cur       -> b_w (MLP acc): {cw:.4f}   (sanity, expect <= 0.58)")

    print("\n  VERDICT:")
    if cx > 0.60 or cs > 0.62 or cw > 0.62:
        print("  BASELINE BROKEN: q-proxy leaks sign. Probe methodology invalid.")
    elif hx_full < 0.55:
        print("  RUN v5.2: hist_full does not contain XOR; aux channel is needed.")
    elif hx_full > 0.70:
        print("  DO NOT RUN v5.2: hist_full contains XOR; GRU_A will learn it without aux.")
    else:
        print("  GREY ZONE: v5.2 launchable with longer t_lock or different feature composition.")

    delta = hx_full - hx_pred
    print(f"\n  Delta(hist_full - hist_pred) for XOR: {delta:+.4f}")
    print("  If Delta > 0.05, downstream channel dx/dy contributes to XOR.")


if __name__ == "__main__":
    main()