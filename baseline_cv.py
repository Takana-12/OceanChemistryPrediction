import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import DEV_SPLIT_SEED, FEATURES, load_data, rmse, save_run

EPOCHS = 5000
X, y, Xte, _ = load_data()
folds = list(KFold(5, shuffle=True, random_state=DEV_SPLIT_SEED).split(X))

if sys.argv[1] == "collect":
    oof = np.zeros(len(y))
    for k, (_, va) in enumerate(folds):
        oof[va] = np.load(f"runs/baseline_fold{k}.npy")
    res = dict(single_rmse=[rmse(oof, y)], ensemble_rmse=rmse(oof, y),
               fold_rmse=[rmse(oof[va], y[va]) for _, va in folds], oofs=oof[None], test_pred=None, curves=[])
    save_run("baseline_5000ep", dict(epochs=EPOCHS, note="配布ノートブックと同じ設定"), res, [0], DEV_SPLIT_SEED)
    print("baseline CV RMSE:", res["ensemble_rmse"], "fold:", np.round(res["fold_rmse"], 3).tolist())
    sys.exit()

torch.set_num_threads(1)
k = int(sys.argv[1])
tr, va = folds[k]
torch.manual_seed(k)
sc = StandardScaler().fit(X[tr])
Xt = torch.tensor(sc.transform(X[tr]), dtype=torch.float32)
yt = torch.tensor(y[tr], dtype=torch.float32)
model = nn.Sequential(nn.Linear(len(FEATURES), 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(EPOCHS):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        nn.functional.mse_loss(model(Xt[idx]).squeeze(-1), yt[idx]).backward()
        opt.step()
with torch.no_grad():
    p = model(torch.tensor(sc.transform(X[va]), dtype=torch.float32)).squeeze(-1).numpy()
np.save(f"runs/baseline_fold{k}.npy", p)
print(f"fold {k}: {rmse(p, y[va]):.3f}", flush=True)
