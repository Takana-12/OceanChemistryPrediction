import sys
import time

import numpy as np
import torch

from common import ALT_SPLIT_SEED, DEV_SPLIT_SEED, cv_run, load_data, save_run

torch.set_num_threads(1)

BASE = dict(hidden=128, depth=2, dropout=0.0, skip=True, lr=3e-3, wd=1e-4, epochs=300, bs=64,
            loss="mse", delta=0.1, clean=None)

EXPS = {
    "mse": {},
    "huber": dict(loss="huber"),
    "mse_clean15": dict(clean=15),
    "huber_clean15": dict(loss="huber", clean=15),
    "h64": dict(hidden=64),
    "h32": dict(hidden=32),
    "wd1e-3": dict(wd=1e-3),
    "wd1e-2": dict(wd=1e-2),
    "ep150": dict(epochs=150),
    "ep600": dict(epochs=600),
    "no_skip": dict(skip=False),
    "huber_h64": dict(loss="huber", hidden=64),
    "huber_ep600": dict(loss="huber", epochs=600),
    "huber_h64_ep600": dict(loss="huber", hidden=64, epochs=600),
}

H = dict(loss="huber")
EXPS.update({
    "hd003": dict(H, delta=0.03), "hd005": dict(H, delta=0.05), "hd02": dict(H, delta=0.2), "hd04": dict(H, delta=0.4),
    "h_lr1e-3": dict(H, lr=1e-3), "h_lr1e-2": dict(H, lr=1e-2),
    "h_bs32": dict(H, bs=32), "h_bs128": dict(H, bs=128),
    "h_wd0": dict(H, wd=0.0), "h_wd1e-3": dict(H, wd=1e-3), "h_wd1e-2": dict(H, wd=1e-2),
    "h_depth1": dict(H, depth=1), "h_depth3": dict(H, depth=3), "h_w256": dict(H, hidden=256),
    "h_relu": dict(H, act="relu"), "h_tanh": dict(H, act="tanh"), "h_gelu": dict(H, act="gelu"),
    "h_drop005": dict(H, dropout=0.05),
    "huber_k10": dict(H, _k=10),
})

B32 = dict(H, bs=32)
EXPS.update({
    "h_bs32_d3": dict(B32, depth=3), "h_bs32_gelu": dict(B32, act="gelu"),
    "h_bs32_d3_gelu": dict(B32, depth=3, act="gelu"), "h_bs32_hd005": dict(B32, delta=0.05),
    "h_bs32_ep600": dict(B32, epochs=600), "h_bs32_h64_ep600": dict(B32, hidden=64, epochs=600),
    "h_bs16": dict(H, bs=16), "h_d3_gelu": dict(H, depth=3, act="gelu"),
})

B32D = dict(H, bs=32, delta=0.05)
EXPS.update({
    "h_bs32_hd003": dict(B32, delta=0.03), "h_bs48_hd005": dict(B32D, bs=48),
    "h_bs32_hd005_ep200": dict(B32D, epochs=200), "h_bs32_hd005_wd1e-2": dict(B32D, wd=1e-2),
})

EXPS.update({
    "h_bs32_hd002": dict(B32, delta=0.02), "h_bs32_hd001": dict(B32, delta=0.01), "h_bs32_hd0003": dict(B32, delta=0.003),
})

SEEDS = (0, 1, 2)

if __name__ == "__main__":
    name = sys.argv[1]
    alt = "--alt-split" in sys.argv
    split_seed = ALT_SPLIT_SEED if alt else DEV_SPLIT_SEED
    custom = "--split-seed" in sys.argv
    if custom:
        split_seed = int(sys.argv[sys.argv.index("--split-seed") + 1])
    cfg = {**BASE, **EXPS[name]}
    n_splits = cfg.pop("_k", 5)
    X, y, Xte, _ = load_data()
    t = time.time()
    res = cv_run(X, y, Xte, cfg, seeds=SEEDS, split_seed=split_seed, n_splits=n_splits, curve=True)
    run_name = name + (f"_split{split_seed}" if custom else f"_alt{ALT_SPLIT_SEED}" if alt else "")
    save_run(run_name, cfg, res, SEEDS, split_seed, extra=dict(n_splits=n_splits))
    print(f"{run_name:16s} 単体 {np.mean(res['single_rmse']):.4f} (各seed {np.round(res['single_rmse'], 3).tolist()})"
          f"  seed平均 {res['ensemble_rmse']:.4f}  ({time.time() - t:.0f}s)", flush=True)
