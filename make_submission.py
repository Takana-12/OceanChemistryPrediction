import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_data, rmse

out = Path(sys.argv[1])
names = sys.argv[2:]
assert out.suffix.lower() == ".csv", "出力ファイルは .csv にする"
meta_path = out.with_name(out.stem + "_meta.json")
assert meta_path.resolve() != out.resolve()

_, y, _, ids = load_data()
metas, preds, oofs = [], [], []
for name in names:
    meta = json.load(open(f"runs/{name}/meta.json", encoding="utf-8"))
    if not meta.get("has_test_pred"):
        sys.exit(f"{name} はテスト予測を計算していないため提出ファイルを作れない")
    metas.append(meta)
    preds.append(np.load(f"runs/{name}/test_pred.npy"))
    oofs.append(np.load(f"runs/{name}/oofs.npy").mean(0))
pred = np.mean(preds, axis=0)
oof_rmse = rmse(np.mean(oofs, axis=0), y)

sub = pd.DataFrame({"id": ids, "DIC": pred})
sample = pd.read_csv("sample_submission.csv")
assert (sub["id"].values == sample["id"].values).all() and np.isfinite(pred).all()
sub.to_csv(out, index=False)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(dict(runs=names, combined_oof_rmse=oof_rmse, run_meta=metas), f, ensure_ascii=False, indent=1)
print(f"{out}: {' + '.join(names)}  OOF RMSE {oof_rmse:.4f}")
