import os
import sys

import numpy as np

from common import ALT_SPLIT_SEED, load_data, rmse

_, y, _, _ = load_data()


def load(name):
    return [np.load(f"runs/{name}{suf}/oofs.npy") for suf in ("", f"_alt{ALT_SPLIT_SEED}")]


ref_name, names = sys.argv[1], sys.argv[2:]
ref = load(ref_name)
ref_mean = (ref[0].mean(0) + ref[1].mean(0)) / 2
bulk = np.ones(len(y), bool)
bulk[np.argsort(-np.abs(ref_mean - y))[:5]] = False

print(f"{'設定':18s} {'seed平均(通常/別)':>18s} {'2分割平均':>9s} {'差':>7s} {'勝ち数':>5s} {'上位5行除外':>10s} {'差':>7s}")
for name in [ref_name] + names:
    if not all(os.path.exists(f"runs/{name}{suf}/oofs.npy") for suf in ("", f"_alt{ALT_SPLIT_SEED}")):
        print(f"{name:18s} (未完了)")
        continue
    o = load(name)
    m = (o[0].mean(0) + o[1].mean(0)) / 2
    wins = sum(rmse(o[i][s], y) < rmse(ref[i][s], y) for i in range(2) for s in range(3))
    print(f"{name:18s} {rmse(o[0].mean(0), y):8.3f} /{rmse(o[1].mean(0), y):7.3f} {rmse(m, y):9.3f} {rmse(m, y) - rmse(ref_mean, y):+7.3f}"
          f" {wins:4d}/6 {rmse(m[bulk], y[bulk]):10.3f} {rmse(m[bulk], y[bulk]) - rmse(ref_mean[bulk], y[bulk]):+7.3f}")
