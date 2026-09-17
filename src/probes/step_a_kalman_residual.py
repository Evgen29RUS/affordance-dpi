# step_a_kalman_residual.py
# Step A: Kalman residual on frozen v4.3 substrate.
# No PPO. No v5.3. Read-only.

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# ------------------- ENV PARAMS (identical to v4.3) -------------------
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

THETA = 0.5; ALPHA = 8.0; GRID = 8; T_HORIZON = 100

PRED_DIM = 4
FULL_DIM = 8
HIST_K   = 8
HIST_DIM_PRED = PRED_DIM * HIST_K + 5 * HIST_K   # 72
HIST_DIM_FULL = FULL_DIM * HIST_K + 5 * HIST_K   # 104

ENV_SEEDS = [6, 7, 8, 9, 10]
DATA_SEED = 999

# uniform(0.25, 0.75) variance
INIT_VAR = (0.75 - 0.25)**2 / 12.0   # 0.020833...


# ------------------- TASKS -------------------
@dataclass(frozen=True)
class TaskCfg:
    task_type: str; distractor: int
    def key(self): return f"{self.task_type}_d{self.distractor}"

def train_tasks():
    keys = {"reach_d0","reach_d1","find_d0","find_d2",
            "avoid_d0","wait_d0","wait_d1"}
    return [TaskCfg(t, d) for t in ["reach","find","avoid","wait"] for d in [0,1,2]
            if f"{t}_d{d}" in keys]


# ------------------- ENVIRONMENT -------------------
class Environment:
    def __init__(self, seed): self.rng = np.random.RandomState(seed)
    def reset(self, cfg, seed):
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg; self.t = 0; self.done = False; self.prev_action = 4
        self.q = self.rng.uniform(0.25, 0.75, 4)
        self.s = self.rng.uniform(0.25, 0.75, 4)
        self.w = self.rng.uniform(0.25, 0.75, 4)
        self.v = self.rng.uniform(0.25, 0.75, 4)
        self.pos = (int(self.rng.randint(0,GRID)), int(self.rng.randint(0,GRID)))
        while True:
            gx=int(self.rng.randint(0,GRID)); gy=int(self.rng.randint(0,GRID))
            if (gx,gy)!=self.pos: break
        self.goal=(gx,gy)
        self.objects=set()
        for _ in range(cfg.distractor):
            while True:
                ox=int(self.rng.randint(0,GRID)); oy=int(self.rng.randint(0,GRID))
                if (ox,oy)!=self.pos and (ox,oy)!=self.goal: break
            self.objects.add((ox,oy))
        self.signal_present=False
        self.signal_timer=int(self.rng.randint(20,60))
        self._compute_obs(); return self.obs

    def _c(self, x, a):
        return float(np.clip(np.dot(M_MAT[:,a], x)/4.0, 0.0, 1.0))
    def _p_exec(self, a):
        z = ALPHA*(self._c(self.s,a)-THETA) + ALPHA*(self._c(self.w,a)-THETA)
        return float(1.0/(1.0+np.exp(-z)))
    def _move(self, a):
        x,y=self.pos
        if a==0: new=(x,max(0,y-1))
        elif a==1: new=(x,min(GRID-1,y+1))
        elif a==2: new=(max(0,x-1),y)
        elif a==3: new=(min(GRID-1,x+1),y)
        else: return True,self.pos
        blocked=(new==self.pos) or (new in self.objects)
        return (not blocked), new

    def step(self, a):
        onehot = np.zeros(5); onehot[a]=1.0
        q_t = self.q.copy()
        es=self.rng.normal(0,SIGMA_X,4)
        ew=self.rng.normal(0,SIGMA_X,4)
        ev=self.rng.normal(0,SIGMA_X,4)
        self.s = np.clip(D_VEC+A_SCALAR*self.s+B_MAT@onehot+C_SCALAR*q_t+es, 0, 1)
        self.w = np.clip(D_VEC+A_SCALAR*self.w+B_MAT@onehot+C_SCALAR*q_t+ew, 0, 1)
        self.v = np.clip(D_VEC+A_SCALAR*self.v+B_MAT@onehot+C_SCALAR*q_t+ev, 0, 1)
        exec_ok = bool(self.rng.rand() < self._p_exec(a))
        if exec_ok:
            ok, np_ = self._move(a)
            if ok: self.pos = np_
        eta=self.rng.normal(0,SIGMA_Q,4)
        self.q = np.clip(RHO_Q*q_t+OFFSET_Q+eta, 0, 1)
        self.signal_timer -= 1
        if self.signal_timer<=0: self.signal_present=True
        self.t += 1
        reward=-0.01; self.done=False
        if self.cfg.task_type in ("reach","find") and self.pos==self.goal:
            reward=1.0; self.done=True
        if self.cfg.task_type=="avoid" and self.pos in self.objects:
            reward=-1.0; self.done=True
        if self.cfg.task_type=="wait" and self.signal_present and self.pos==self.goal:
            reward=1.0; self.done=True
        if self.t>=T_HORIZON: self.done=True
        self._compute_obs(); return self.obs, reward, self.done

    def _compute_obs(self):
        nu=self.rng.normal(0,SIGMA_OBS,4)
        yq=np.clip(self.q+nu,0,1)
        x,yy=self.pos; gx,gy=self.goal
        dx=(gx-x)/GRID; dy=(gy-yy)/GRID
        if self.objects:
            dmin=min(abs(ox-x)+abs(oy-yy) for ox,oy in self.objects)/(2*GRID)
        else: dmin=1.0
        sig=1.0 if self.signal_present else 0.0
        self.obs=np.concatenate([yq,[dx,dy,dmin,sig]]).astype(np.float64)


# ------------------- DATA COLLECTION (per-episode) -------------------
def flatten_hist(obs_h, act_h, dim):
    while len(obs_h) < HIST_K:
        obs_h = [np.zeros(dim, dtype=np.float32)] + obs_h
        act_h = [4] + act_h
    obs_h = obs_h[-HIST_K:]; act_h = act_h[-HIST_K:]
    a_oh = []
    for a in act_h:
        v = np.zeros(5, dtype=np.float32); v[a]=1.0; a_oh.append(v)
    return np.concatenate([np.concatenate(obs_h),
                           np.concatenate(a_oh)]).astype(np.float32)


def collect_episodes(tasks, seeds, n_ep=8, data_seed=DATA_SEED):
    env = Environment(seed=42)
    rng = np.random.RandomState(data_seed)
    episodes = []
    tid = 0
    for cfg in tasks:
        for sd in seeds:
            for _ in range(n_ep):
                o = env.reset(cfg, sd)
                done = False
                yqs   = [o[:PRED_DIM].copy()]
                obs_f = [o.copy()]
                s_seq = [env.s.copy()]
                w_seq = [env.w.copy()]
                v_seq = [env.v.copy()]
                acts  = []
                hist_o_pred, hist_o_full, hist_a = [], [], []
                H_pred_list, H_full_list = [], []
                while not done:
                    a = int(rng.randint(0, 5))
                    o_pred = yqs[-1]; o_full = obs_f[-1]
                    hp = flatten_hist(hist_o_pred + [o_pred], hist_a + [a], PRED_DIM)
                    hf = flatten_hist(hist_o_full + [o_full], hist_a + [a], FULL_DIM)
                    H_pred_list.append(hp); H_full_list.append(hf)
                    acts.append(a)
                    o_next, _, done = env.step(a)
                    yqs.append(o_next[:PRED_DIM].copy())
                    obs_f.append(o_next.copy())
                    s_seq.append(env.s.copy())
                    w_seq.append(env.w.copy())
                    v_seq.append(env.v.copy())
                    hist_o_pred.append(o_pred); hist_o_full.append(o_full); hist_a.append(a)
                    if len(hist_a) > HIST_K:
                        hist_o_pred = hist_o_pred[-HIST_K:]
                        hist_o_full = hist_o_full[-HIST_K:]
                        hist_a = hist_a[-HIST_K:]
                T = len(acts)
                episodes.append({
                    "y_q":     np.array(yqs[:-1]),         # (T,4)
                    "a":       np.array(acts),             # (T,)
                    "s":       np.array(s_seq),            # (T+1,4) including s_0
                    "w":       np.array(w_seq),
                    "v":       np.array(v_seq),
                    "H_pred":  np.array(H_pred_list),      # (T,72)
                    "H_full":  np.array(H_full_list),      # (T,104)
                    "cur":     np.array(yqs[:-1]),         # (T,4) — same as y_q(t)
                    "tid":     tid,
                })
                tid += 1
    return episodes


# ------------------- KALMAN FILTER (8-dim: s, q) -------------------
def kalman_residual_episode(ep):
    """
    Returns per-step:
      r_s: (T,4)   residual s_{t+1} - mu_{s,t+1|t}
      P_s: (T,4,4) predicted covariance for s-component
      clipped: (T,4) per-dim clip indicator at observation y_q(t)
    """
    T = ep["y_q"].shape[0]
    I4 = np.eye(4); Z4 = np.zeros((4,4)); I8 = np.eye(8)
    F = np.block([[A_SCALAR*I4, C_SCALAR*I4],
                  [Z4,           RHO_Q*I4]])
    d = np.concatenate([D_VEC, OFFSET_Q*np.ones(4)])
    G = np.vstack([B_MAT, np.zeros((4,5))])
    Q = np.diag(np.concatenate([SIGMA_X**2*np.ones(4),
                                SIGMA_Q**2*np.ones(4)]))
    H = np.hstack([Z4, I4])
    R = SIGMA_OBS**2 * I4

    # prior on x_0 = (s_0, q_0)
    mu = np.concatenate([0.5*np.ones(4), 0.5*np.ones(4)])
    Sigma = INIT_VAR * I8

    r_s   = np.zeros((T,4))
    P_s   = np.zeros((T,4,4))
    clip  = np.zeros((T,4), dtype=bool)

    for t in range(T):
        y = ep["y_q"][t]
        clip[t] = (y <= 1e-9) | (y >= 1 - 1e-9)

        # measurement update with y_q(t)
        S = H @ Sigma @ H.T + R
        K = Sigma @ H.T @ np.linalg.inv(S)
        mu = mu + K @ (y - H @ mu)
        Sigma = (I8 - K @ H) @ Sigma

        # prediction (x_t -> x_{t+1})
        onehot = np.zeros(5); onehot[ep["a"][t]] = 1.0
        mu_pred = F @ mu + d + G @ onehot
        Sigma_pred = F @ Sigma @ F.T + Q

        # residual against true s_{t+1}
        r_s[t] = ep["s"][t+1] - mu_pred[:4]
        P_s[t] = Sigma_pred[:4, :4]

        # advance
        mu = mu_pred
        Sigma = Sigma_pred

    return r_s, P_s, clip


# ------------------- PROBES (same structure as classify_probe) -------------------
class LinProbe(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 2)
    def forward(self, x): return self.fc(x)

class MLPProbe(nn.Module):
    def __init__(self, d, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,hid), nn.ReLU(),
                                 nn.Linear(hid,hid), nn.ReLU(),
                                 nn.Linear(hid,2))
    def forward(self, x): return self.net(x)

def train_probe(Xtr, ytr, Xva, yva, model, seed, n_steps=4000, lr=1e-3):
    torch.manual_seed(seed)
    Xt=torch.tensor(Xtr); yt=torch.tensor(ytr, dtype=torch.long)
    Xv=torch.tensor(Xva); yv=torch.tensor(yva, dtype=torch.long)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    best_acc=-1.0; best_state=None
    for step in range(n_steps):
        model.train()
        idx=torch.randperm(Xt.shape[0])[:256]
        loss=nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                acc=(model(Xv).argmax(1)==yv).float().mean().item()
            if acc>best_acc:
                best_acc=acc
                best_state={k:v.clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    yv_np = yva
    acc = float((pred==yv_np).mean())
    baccs=[]
    for c in [0,1]:
        m = yv_np==c
        if m.sum()>0: baccs.append(float((pred[m]==c).mean()))
    return acc, float(np.mean(baccs)) if baccs else float("nan")

def evaluate_probe(X, y, traj_id, model_ctor, n_folds=5, base_seed=0):
    u = np.unique(traj_id); rng=np.random.RandomState(base_seed); rng.shuffle(u)
    folds = np.array_split(u, n_folds)
    accs, baccs = [], []
    for k, va_t in enumerate(folds):
        va = np.isin(traj_id, va_t); tr = ~va
        if tr.sum()==0 or va.sum()==0: continue
        acc, bacc = train_probe(X[tr], y[tr], X[va], y[va],
                                model_ctor(X.shape[1]), seed=base_seed*1000+k)
        accs.append(acc); baccs.append(bacc)
    return float(np.mean(accs)), float(np.mean(baccs))

def train_regressor(Xtr, Ytr, Xva, Yva, seed, in_dim, out_dim=4, n_steps=3000, lr=1e-3):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(),
                          nn.Linear(64,64), nn.ReLU(),
                          nn.Linear(64,out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt=torch.tensor(Xtr, dtype=torch.float32)
    Yt=torch.tensor(Ytr, dtype=torch.float32)
    Xv=torch.tensor(Xva, dtype=torch.float32)
    Yv=torch.tensor(Yva, dtype=torch.float32)
    best=float("inf"); best_state=None; since=0; patience=500
    for step in range(n_steps):
        idx=torch.randperm(Xt.shape[0])[:256]
        loss=((model(Xt[idx])-Yt[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0:
            with torch.no_grad():
                vl=((model(Xv)-Yv)**2).mean().item()
            if vl < best - 1e-6:
                best=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}
                since=0
            else:
                since+=100
            if since>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).numpy()
    mse = float(np.mean((pred - Yva)**2))
    var = float(np.mean((Yva - Yva.mean(0))**2))
    return mse, var, mse/max(var,1e-12)


# ------------------- MAIN -------------------
def main():
    print("="*78)
    print("STEP A: Kalman residual on frozen v4.3 substrate")
    print("="*78)

    tasks = train_tasks()
    eps_list = collect_episodes(tasks, ENV_SEEDS, n_ep=8)
    n_samples = sum(e["y_q"].shape[0] for e in eps_list)
    print(f"episodes: {len(eps_list)}   samples: {n_samples}")

    # ----- per-episode Kalman -----
    R_s_all   = []
    P_s_all   = []
    clip_all  = []
    tids      = []
    H_pred    = []
    H_full    = []
    CUR       = []
    S_NEXT    = []
    W_NEXT    = []
    V_NEXT    = []

    for ep in eps_list:
        r_s, P_s, clip = kalman_residual_episode(ep)
        R_s_all.append(r_s)
        P_s_all.append(P_s)
        clip_all.append(clip)
        tids.extend([ep["tid"]] * ep["y_q"].shape[0])
        H_pred.append(ep["H_pred"])
        H_full.append(ep["H_full"])
        CUR.append(ep["cur"])
        S_NEXT.append(ep["s"][1:])
        W_NEXT.append(ep["w"][1:])
        V_NEXT.append(ep["v"][1:])

    R_s   = np.concatenate(R_s_all, axis=0)
    P_s   = np.concatenate(P_s_all, axis=0)
    CLIP  = np.concatenate(clip_all, axis=0)
    H_pred= np.concatenate(H_pred, axis=0)
    H_full= np.concatenate(H_full, axis=0)
    CUR   = np.concatenate(CUR, axis=0)
    S_NEXT= np.concatenate(S_NEXT, axis=0)
    tids  = np.array(tids)

    # ----- P-1a: analytical residual statistics (pre-clip) -----
    print("\n--- P-1a: residual statistics (analytical Kalman, pre-clip) ---")
    mean_r = R_s.mean(axis=0)
    var_r  = R_s.var(axis=0)
    mean_P = np.array([np.diag(P_s[t]).mean() for t in range(P_s.shape[0])])
    # actually: mean over t of diag(P_s[t])[k]
    mean_P = np.diag(np.mean(P_s, axis=0))
    ratio  = var_r / np.maximum(mean_P, 1e-12)
    norm_bias = np.abs(mean_r) / np.sqrt(np.maximum(mean_P, 1e-12))
    for k in range(4):
        print(f"  dim{k}: mean_r={mean_r[k]:+.5f}  var_r={var_r[k]:.5f}  "
              f"mean_P={mean_P[k]:.5f}  var/mean_P={ratio[k]:.3f}  "
              f"|bias|/sqrt(P)={norm_bias[k]:.4f}")
    print(f"  max |bias|/sqrt(P) = {norm_bias.max():.4f}   "
          f"(PASS threshold < 0.10)")

    # ----- P-1b: clip diagnostics -----
    print("\n--- P-1b: clip diagnostics ---")
    frac_clip_per_dim = CLIP.mean(axis=0)
    frac_clip_any    = CLIP.any(axis=1).mean()
    for k in range(4):
        print(f"  dim{k}: fraction clipped = {frac_clip_per_dim[k]:.4f}")
    print(f"  any-dim clipped: {frac_clip_any:.4f}")
    # correlation of |r_s| with clip (rough)
    abs_r = np.abs(R_s).mean(axis=1)
    clip_any = CLIP.any(axis=1).astype(float)
    if clip_any.std() > 0:
        corr_clip = float(np.corrcoef(abs_r, clip_any)[0,1])
        print(f"  corr(|r_s|_mean, any_clip) = {corr_clip:+.4f}")

    # ----- P-2: linear orthogonality of residual w.r.t. history -----
    print("\n--- P-2: linear orthogonality ---")
    # features: current y_q, one-hot of last 8 actions, constant
    feats = []
    for t in range(R_s.shape[0]):
        pass  # slow if python loop; use vectorized construction instead
    # vectorized: use H_pred already contains past y_q + past actions.
    # correlate residual against each column of H_pred, take max |corr|
    max_corr = 0.0
    for k in range(4):
        yk = R_s[:, k]
        if yk.std() < 1e-9: continue
        # sample H_pred columns to save time
        cols = H_pred.shape[1]
        cs = np.zeros(cols)
        for j in range(cols):
            xj = H_pred[:, j]
            if xj.std() < 1e-9: continue
            cs[j] = abs(np.corrcoef(xj, yk)[0,1])
        mx = cs.max()
        print(f"  dim{k}: max |corr| over {cols} H_pred features = {mx:.5f}")
        max_corr = max(max_corr, mx)
    print(f"  overall max |corr| = {max_corr:.5f}   (PASS threshold < 0.05)")

    # ----- P-3: classification probes -----
    # define target: sign(r_s[0] - r_s[1])
    r_lin = R_s[:,0] - R_s[:,1]
    y_sign = (r_lin > 0).astype(np.int64)
    print(f"\n--- P-3: classification probes on sign(r_s[0]-r_s[1]) ---")
    print(f"  P(sign=1) = {y_sign.mean():.4f}   (balance check)")

    # for reference: sign(s_next[0]-s_next[1])
    s_lin = S_NEXT[:,0] - S_NEXT[:,1]
    y_sign_s = (s_lin > 0).astype(np.int64)
    print(f"  P(sign(s_next)=1) = {y_sign_s.mean():.4f}   (for reference)")

    rows = []
    for name_in, X in [("H_pred", H_pred), ("H_full", H_full), ("CUR", CUR)]:
        a_lin, ba_lin = evaluate_probe(X, y_sign, tids, LinProbe,  base_seed=1)
        a_mlp, ba_mlp = evaluate_probe(X, y_sign, tids, MLPProbe, base_seed=2)
        print(f"  {name_in:8s} -> sign(r_s) | "
              f"lin acc={a_lin:.4f} bacc={ba_lin:.4f} | "
              f"mlp acc={a_mlp:.4f} bacc={ba_mlp:.4f}")
        rows.append((name_in, a_lin, ba_lin, a_mlp, ba_mlp))

    # s_next -> sign(r_s) sanity (should be ~1)
    a_lin_s, ba_lin_s = evaluate_probe(S_NEXT, y_sign, tids, LinProbe,  base_seed=3)
    a_mlp_s, ba_mlp_s = evaluate_probe(S_NEXT, y_sign, tids, MLPProbe, base_seed=4)
    print(f"  S_next   -> sign(r_s) | "
          f"lin acc={a_lin_s:.4f} bacc={ba_lin_s:.4f} | "
          f"mlp acc={a_mlp_s:.4f} bacc={ba_mlp_s:.4f}   (sanity, expect ~1)")

    # ----- P-4: R² of MLP(H_pred -> r_s) -----
    print("\n--- P-4: MLP(H_pred) -> r_s, out-of-sample R² ---")
    n = R_s.shape[0]
    idx = np.random.RandomState(2024).permutation(n)
    tr = idx[: n*3//4]; va = idx[n*3//4:]
    mse, var, r2 = train_regressor(H_pred[tr], R_s[tr], H_pred[va], R_s[va],
                                   seed=99, in_dim=H_pred.shape[1], out_dim=4)
    print(f"  OOS MSE = {mse:.6f}   Var(r_s) = {var:.6f}   R² = {1.0 - mse/max(var,1e-12):.5f}")
    print(f"  (PASS threshold: R² ≤ 0.02)")

    # ----- final verdict -----
    print("\n" + "="*78)
    print("VERDICT (preregistered)")
    print("="*78)
    p1a_pass = bool(norm_bias.max() < 0.10)
    p1a_var_ok = bool(np.all((ratio > 0.7) & (ratio < 1.5)))
    p2_pass = bool(max_corr < 0.05)
    p3_hist = next(r for r in rows if r[0]=="H_full")
    p3_pass = bool(p3_hist[3] < 0.55)     # mlp bacc
    p4_pass = bool(r2 <= 0.02)

    print(f"  P-1a (bias < 0.10):        {p1a_pass}")
    print(f"  P-1a (var/mean_P in range): {p1a_var_ok}   ratios = "
          f"{np.array2string(ratio, precision=3)}")
    print(f"  P-1b clip fraction (any):   {frac_clip_any:.4f}   "
          f"(diagnostic only)")
    print(f"  P-2  (max |corr| < 0.05):   {p2_pass}   max = {max_corr:.5f}")
    print(f"  P-3  (H_full → sign < 0.55): {p3_pass}   "
          f"MLP bacc = {p3_hist[3]:.4f}")
    print(f"  P-4  (R² ≤ 0.02):            {p4_pass}   R² = {1.0 - mse/max(var,1e-12):.5f}")

    all_pass = p1a_pass and p1a_var_ok and p2_pass and p3_pass and p4_pass
    print(f"\n  ALL PASS: {all_pass}")

    if all_pass:
        print("  Residual is history-independent. Target candidate valid.")
        print("  NOTE: this does NOT imply aux can carry it. Step B still blocked")
        print("        on §4 information-flow question.")
    elif not (p1a_pass and p1a_var_ok):
        print("  Kalman model mismatch. Clip may explain. Check P-1b.")
    elif not p2_pass or not p3_pass or not p4_pass:
        print("  Residual is NOT history-independent. Kalman information set")
        print("  may not match agent's, or non-linearity in p_exec leaks.")


if __name__ == "__main__":
    main()