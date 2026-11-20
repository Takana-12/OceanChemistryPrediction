import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = "."
RUNS_DIR = "runs"

FEATURES = [
    "Lat_Dec", "Lon_Dec", "NO2uM", "NO3uM", "NH3uM", "R_TEMP", "R_Depth", "R_Sal",
    "R_DYNHT", "R_Nuts", "R_Oxy_micromol.Kg", "PO4uM", "SiO3uM", "Salinity1", "Temperature_degC",
]

DEV_SPLIT_SEED = 42
ALT_SPLIT_SEED = 2026


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    common = train.drop(columns=["Unnamed: 12", "id", "DIC"]).columns.intersection(test.columns)
    assert list(common) == FEATURES, f"入力列がベースラインと一致しない: {list(common)}"
    return train[FEATURES].values, train["DIC"].values, test[FEATURES].values, test["id"].values


class Net(nn.Module):

    def __init__(self, d_in, hidden=128, depth=2, dropout=0.0, skip=True, act="silu"):
        super().__init__()
        act_cls = dict(silu=nn.SiLU, relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU)[act]
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), act_cls(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.skip = nn.Linear(d_in, 1) if skip else None

    def forward(self, x):
        out = self.mlp(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out.squeeze(-1)


def fit_predict(X_tr, y_tr, X_preds, cfg, seed, curve=None):
    if cfg.get("clean"):
        (p_tr,) = fit_predict(X_tr, y_tr, [X_tr], {**cfg, "clean": None}, seed)
        keep = np.abs(p_tr - y_tr) < cfg["clean"]
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        cfg = {**cfg, "clean": None}
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs = StandardScaler().fit(X_tr)
    y_mu, y_sd = y_tr.mean(), y_tr.std()
    X = torch.tensor(xs.transform(X_tr), dtype=torch.float32)
    y = torch.tensor((y_tr - y_mu) / y_sd, dtype=torch.float32)

    model = Net(X.shape[1], cfg["hidden"], cfg["depth"], cfg["dropout"], cfg["skip"], cfg.get("act", "silu"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs, bs = cfg["epochs"], cfg["bs"]
    n = len(X)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.HuberLoss(delta=cfg["delta"])

    def predict(Xp):
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(xs.transform(Xp), dtype=torch.float32)).numpy()
        model.train()
        return p * y_sd + y_mu

    history = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            sched.step()
        if curve is not None and (ep + 1) % 10 == 0:
            history.append(float(np.sqrt(np.mean((predict(curve[0]) - curve[1]) ** 2))))

    outs = [predict(Xp) for Xp in X_preds]
    return (outs, history) if curve is not None else outs


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv_run(X, y, X_test, cfg, seeds=(0,), split_seed=DEV_SPLIT_SEED, n_splits=5, curve=False):
    folds = list(KFold(n_splits, shuffle=True, random_state=split_seed).split(X))
    oofs = np.zeros((len(seeds), len(y)))
    test_pred = np.zeros(len(X_test))
    curves = []
    for si, s in enumerate(seeds):
        for k, (tr, va) in enumerate(folds):
            out = fit_predict(X[tr], y[tr], [X[va], X_test], cfg, seed=1000 * s + k,
                              curve=(X[va], y[va]) if curve else None)
            if curve:
                out, hist = out
                curves.append(hist)
            oofs[si, va], p_te = out
            test_pred += p_te / (len(seeds) * n_splits)
    oof_mean = oofs.mean(0)
    return dict(
        single_rmse=[rmse(o, y) for o in oofs],
        ensemble_rmse=rmse(oof_mean, y),
        fold_rmse=[rmse(oof_mean[va], y[va]) for _, va in folds],
        oofs=oofs, test_pred=test_pred, curves=curves,
    )


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def save_run(name, cfg, res, seeds, split_seed, extra=None):
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "oofs.npy"), res["oofs"])
    has_test_pred = res["test_pred"] is not None
    if has_test_pred:
        np.save(os.path.join(d, "test_pred.npy"), res["test_pred"])
    meta = dict(
        name=name, cfg=cfg, has_test_pred=has_test_pred, features=FEATURES, seeds=list(seeds), split_seed=split_seed,
        single_rmse=res["single_rmse"], single_rmse_mean=float(np.mean(res["single_rmse"])),
        ensemble_rmse=res["ensemble_rmse"], fold_rmse=res["fold_rmse"],
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        hashes={f: _sha256(f) for f in ["train.csv", "test.csv", "common.py", "experiments.py"]},
        time=time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {}),
    )
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if res["curves"]:
        np.save(os.path.join(d, "curves.npy"), np.array(res["curves"]))
    return meta
