# g4_permutation_sigma010.py
# Same as g4_permutation.py, but loads checkpoints from sigma010 dir.

import os
import numpy as np
import torch
import json

from v5_3_train_sigma010 import (
    Agent, reset_env, variant_channel, bacc, SEEDS,
)

CHECKPOINT_DIR = 'v5_3_checkpoints_sigma010'
RESULT_FILE    = 'v5_3_g4_results_sigma010.json'
G4_N    = 2000
G4_SEED = 777
VARIANTS = ['B', 'C', 'D']

CHANNEL_KEY = {'B': 'c_u', 'C': 'c_w', 'D': 'c_v'}
OWN_BIT     = {'B': 0, 'C': 1, 'D': None}


def load_agent(variant, seed):
    path = os.path.join(CHECKPOINT_DIR, f'{variant}_{seed}.pt')
    if not os.path.exists(path):
        return None
    agent = Agent(variant)
    agent.load_state_dict(torch.load(path, map_location='cpu'))
    agent.eval()
    return agent


def make_eval_set(seed, n_ep=G4_N):
    rng = np.random.RandomState(seed)
    o_l, cu_l, cw_l, cv_l, zu_l, zw_l = [], [], [], [], [], []
    for _ in range(n_ep):
        env = reset_env(rng)
        o_l.append(env['o'])
        cu_l.append(env['c_u'])
        cw_l.append(env['c_w'])
        cv_l.append(env['c_v'])
        zu_l.append(env['z_u'])
        zw_l.append(env['z_w'])
    return {
        'o':  np.stack(o_l).astype(np.float32),
        'c_u': np.stack(cu_l).astype(np.float32),
        'c_w': np.stack(cw_l).astype(np.float32),
        'c_v': np.stack(cv_l).astype(np.float32),
        'z_u': np.array(zu_l),
        'z_w': np.array(zw_l),
    }


def forward(agent, o_t, c_t):
    with torch.no_grad():
        out = agent(o_t, c_t)
        return out['logits'].numpy(), out['aux'].numpy()


def bacc_bit(a_np, z, bit):
    pred = (a_np // 2) if bit == 0 else (a_np % 2)
    return bacc(pred, z)


def permute(arr, seed):
    perm = np.random.RandomState(seed).permutation(arr.shape[0])
    return arr[perm]


def run_one(variant, seed):
    agent = load_agent(variant, seed)
    if agent is None:
        return None

    ev = make_eval_set(seed)
    o_t = torch.tensor(ev['o'])

    c_key = CHANNEL_KEY[variant]
    c_normal = ev[c_key]
    c_perm   = permute(c_normal, seed=seed * 1000 + 42)

    logits_n, aux_n = forward(agent, o_t, torch.tensor(c_normal))
    logits_p, aux_p = forward(agent, o_t, torch.tensor(c_perm))

    a_n = logits_n.argmax(-1)
    a_p = logits_p.argmax(-1)

    result = {'variant': variant, 'seed': seed}
    result['normal_a_u_z_u'] = bacc_bit(a_n, ev['z_u'], 0)
    result['normal_a_w_z_w'] = bacc_bit(a_n, ev['z_w'], 1)
    result['perm_a_u_z_u']   = bacc_bit(a_p, ev['z_u'], 0)
    result['perm_a_w_z_w']   = bacc_bit(a_p, ev['z_w'], 1)
    result['normal_aux_u_z_u'] = bacc((aux_n[:, 0] > aux_n[:, 1]).astype(int), ev['z_u'])
    result['normal_aux_w_z_w'] = bacc((aux_n[:, 0] > aux_n[:, 1]).astype(int), ev['z_w'])
    result['perm_aux_u_z_u']   = bacc((aux_p[:, 0] > aux_p[:, 1]).astype(int), ev['z_u'])
    result['perm_aux_w_z_w']   = bacc((aux_p[:, 0] > aux_p[:, 1]).astype(int), ev['z_w'])

    own = OWN_BIT[variant]
    if own == 0:
        result['own_bit'] = 'u'
        result['own_normal'] = result['normal_a_u_z_u']
        result['own_perm']   = result['perm_a_u_z_u']
    elif own == 1:
        result['own_bit'] = 'w'
        result['own_normal'] = result['normal_a_w_z_w']
        result['own_perm']   = result['perm_a_w_z_w']
    else:
        result['own_bit'] = 'u'
        result['own_normal'] = result['normal_a_u_z_u']
        result['own_perm']   = result['perm_a_u_z_u']
    result['own_drop'] = result['own_normal'] - result['own_perm']
    return result


def main():
    print("=" * 78)
    print("G4: permutation control at SIGMA_C = 0.010")
    print("=" * 78)
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Result file:    {RESULT_FILE}")
    print(f"Eval n per (variant, seed): {G4_N}")
    print(f"Preregistered criterion: permuted own-bit bacc <= 0.55\n")

    results = []
    for variant in VARIANTS:
        print(f"--- {variant} ---")
        for seed in SEEDS:
            r = run_one(variant, seed)
            if r is None:
                print(f"  seed {seed}: checkpoint missing")
                continue
            results.append(r)
            own = r['own_bit']
            print(f"  seed {seed}: "
                  f"{own}-bit normal={r['own_normal']:.4f}  "
                  f"perm={r['own_perm']:.4f}  "
                  f"drop={r['own_drop']:+.4f}  "
                  f"| aux own: normal={r[f'normal_aux_{own}_z_{own}']:.4f}  "
                  f"perm={r[f'perm_aux_{own}_z_{own}']:.4f}")
        print()

    if not results:
        print("No checkpoints found.")
        return

    print("=" * 78)
    print("AGGREGATE (median [min, max] over seeds)")
    print("=" * 78)

    by_v = {v: [r for r in results if r['variant'] == v] for v in VARIANTS}
    for v in VARIANTS:
        rs = by_v[v]
        if not rs: continue
        own_n = np.array([r['own_normal'] for r in rs])
        own_p = np.array([r['own_perm']   for r in rs])
        drop  = np.array([r['own_drop']   for r in rs])
        print(f"{v:>4} ({rs[0]['own_bit']}): "
              f"normal {np.median(own_n):.4f} "
              f"[{own_n.min():.4f},{own_n.max():.4f}]  "
              f"perm {np.median(own_p):.4f} "
              f"[{own_p.min():.4f},{own_p.max():.4f}]  "
              f"drop {np.median(drop):+.4f}")

    print("\n" + "=" * 78)
    print("PREREGISTERED G4 CRITERION")
    print("=" * 78)
    threshold = 0.55
    print(f"  Own-bit permuted bacc <= {threshold}\n")

    b_perm = np.array([r['own_perm'] for r in by_v['B']])
    c_perm = np.array([r['own_perm'] for r in by_v['C']])
    d_perm = np.array([r['own_perm'] for r in by_v['D']])

    b_pass = bool(b_perm.max() <= threshold)
    c_pass = bool(c_perm.max() <= threshold)
    d_pass = bool(d_perm.max() <= threshold)

    print(f"  B: perm max = {b_perm.max():.4f}  {'PASS' if b_pass else 'FAIL'}")
    print(f"  C: perm max = {c_perm.max():.4f}  {'PASS' if c_pass else 'FAIL'}")
    print(f"  D: perm max = {d_perm.max():.4f}  {'PASS' if d_pass else 'FAIL'}")

    b_drop = np.array([r['own_drop'] for r in by_v['B']])
    c_drop = np.array([r['own_drop'] for r in by_v['C']])
    print(f"\n  B: median drop = {np.median(b_drop):+.4f}")
    print(f"  C: median drop = {np.median(c_drop):+.4f}")

    all_pass = b_pass and c_pass and d_pass
    print(f"\n  G4 (SIGMA_C=0.010): {'PASS' if all_pass else 'FAIL'}")

    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'config': {'SIGMA_C': 0.010, 'eval_n': G4_N, 'eval_seed': G4_SEED,
                       'threshold': threshold, 'variants': VARIANTS, 'seeds': SEEDS},
            'per_seed': results,
            'aggregate': {
                'B_perm_max': float(b_perm.max()),
                'C_perm_max': float(c_perm.max()),
                'D_perm_max': float(d_perm.max()),
                'B_drop_median': float(np.median(b_drop)),
                'C_drop_median': float(np.median(c_drop)),
                'pass': bool(all_pass),
            },
        }, f, indent=2, default=str)
    print(f"\nSaved to {RESULT_FILE}")


if __name__ == '__main__':
    main()