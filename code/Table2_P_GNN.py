"""
Table2_P_GNN_loo.py
====================
Leave-One-Out STGNN for wind turbines.
- Trains on all turbines except target (2017 data)
- Tests on target for 2017 and 2018
- Package-friendly paths (resolves from cwd or parent)
- Incremental save + resume
- Year argument: python code/Table2_P_GNN_loo.py 2017
"""

import os, sys, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# =========================
# PATHS
# =========================
ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR     = ROOT / "data"
RESULTS_DIR  = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_CSV = DATA_DIR / "location.csv"
DETAIL_FILE  = RESULTS_DIR / "Table2_P_GNN_loo_detail.csv"
SUMMARY_FILE = RESULTS_DIR / "Table2_P_GNN_loo_summary.csv"

# =========================
# YEAR ARGUMENT
# =========================
args       = sys.argv[1:]
TEST_YEARS = [int(args[0])] if args else [2017, 2018]

# =========================
# CONFIG
# =========================
TURBINE_IDS  = list(range(1, 67))
EXCLUDE_2018 = {47, 51, 53, 61}

TIME_COL   = "time_stamp"
TARGET_COL = "power"
BASE_FEATS = ["wind_speed", "wind_direction", "temperature", "turbulence_intensity", "std_wind_direction"]
FEAT_COLS  = ["wind_speed", "wind_direction_sin", "wind_direction_cos",
              "temperature", "turbulence_intensity", "std_wind_direction"]

LOOKBACK_T  = 24
FREQ_MIN    = 10
K_NEIGHBORS = 7
EDGE_K      = 7

SEED       = 15
EPOCHS     = 10
BATCH_SIZE = 512
LR         = 0.000943
LR_DECAY   = 0.990
MAX_TRAIN_WINDOWS_PER_TURBINE = 1000
NUM_WORKERS = 0
TRAIN_ALLOW_IMPUTE = False

ENCODING_CHANNELS = 64
HIDDEN_CHANNELS   = 64
NUM_MP_LAYERS     = 2
NUM_LSTM_LAYERS   = 2
GNN_DROPOUT       = 0.0896
ENC_DROPOUT       = 0.0916

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# LOCATIONS
# =========================
def load_locations():
    loc = pd.read_csv(LOCATION_CSV)
    loc.columns = [c.strip() for c in loc.columns]
    loc = loc.rename(columns={loc.columns[0]: "Turbine"})
    nm = [c.lower() for c in loc.columns]
    lon_col = loc.columns[[i for i,c in enumerate(nm) if c in ["longitude","lon","long","x"]][0]]
    lat_col = loc.columns[[i for i,c in enumerate(nm) if c in ["latitude","lat","y"]][0]]
    loc[lon_col] = pd.to_numeric(loc[lon_col], errors="coerce")
    loc[lat_col] = pd.to_numeric(loc[lat_col], errors="coerce")
    return loc, lon_col, lat_col

LOC, LON_COL, LAT_COL = load_locations()
POS = {int(r["Turbine"]): (float(r[LON_COL]), float(r[LAT_COL])) for _, r in LOC.iterrows()}

ALL_IDS   = sorted(TURBINE_IDS)
ID_TO_IDX = {tid: i for i, tid in enumerate(ALL_IDS)}
POS_ARR   = np.array([POS[tid] for tid in ALL_IDS], dtype=np.float64)
DX   = POS_ARR[:,None,0] - POS_ARR[None,:,0]
DY   = POS_ARR[:,None,1] - POS_ARR[None,:,1]
DIST = np.sqrt(DX**2 + DY**2) + 1e-12
ANG  = np.arctan2(-DY, -DX)

# =========================
# HELPERS
# =========================
def read_turbine_year(tid, year):
    f = DATA_DIR / f"Turbine{tid}_{year}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)

def preprocess_df(df):
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    for c in BASE_FEATS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rad = df["wind_direction"] * (math.pi / 180.0)
    df["wind_direction_sin"] = np.sin(rad)
    df["wind_direction_cos"] = np.cos(rad)
    keep = df[FEAT_COLS + [TARGET_COL]].notna().all(axis=1)
    df = df.loc[keep, [TIME_COL] + FEAT_COLS + [TARGET_COL]].reset_index(drop=True)
    df[TARGET_COL] = np.clip(df[TARGET_COL].values.astype(np.float32) / 100.0, 0.0, 1.0)
    return df

def compute_scaler(train_data):
    X = np.vstack([df[FEAT_COLS].values.astype(np.float32) for df in train_data.values()])
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd

def standardize(df, mu, sd):
    return ((df[FEAT_COLS].values.astype(np.float32) - mu) / sd).astype(np.float32)

def knn_ids_within_set(focus_tid, candidate_tids, k):
    focus_idx = ID_TO_IDX[focus_tid]
    cand = [t for t in candidate_tids if t != focus_tid]
    if not cand:
        return [focus_tid]
    cand_idxs = np.array([ID_TO_IDX[t] for t in cand], dtype=int)
    order = np.argsort(DIST[focus_idx, cand_idxs])[:min(k, len(cand))]
    return [focus_tid] + [cand[i] for i in order]

def build_edges_for_node_list(node_ids, k=EDGE_K):
    n = len(node_ids)
    if n < 2:
        return torch.empty((2,0), dtype=torch.long), torch.empty((0,3), dtype=torch.float32)
    idxs = np.array([ID_TO_IDX[tid] for tid in node_ids], dtype=int)
    subD = DIST[np.ix_(idxs, idxs)]
    subA = ANG[np.ix_(idxs, idxs)]
    src, dst, eattr = [], [], []
    kk = min(k, n-1)
    for i in range(n):
        order = np.argsort(subD[i])
        nn = [j for j in order if j != i][:kk]
        for j in nn:
            src.append(i); dst.append(j)
            eattr.append([float(subD[i,j]), math.sin(float(subA[i,j])), math.cos(float(subA[i,j]))])
    if not eattr:
        return torch.empty((2,0), dtype=torch.long), torch.empty((0,3), dtype=torch.float32)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(eattr, dtype=torch.float32)
    dd = edge_attr[:,0]
    edge_attr[:,0] = (dd - dd.mean()) / (dd.std() + 1e-6)
    return edge_index, edge_attr

def _nearest_index(tvals, t):
    j = int(np.searchsorted(tvals, t))
    if j <= 0: return 0
    if j >= len(tvals): return len(tvals)-1
    return j if abs(tvals[j]-t) < abs(t-tvals[j-1]) else j-1

def get_lag_indices_impute(tvals, center_t, step_ns, lookback_t):
    return [_nearest_index(tvals, np.int64(center_t - (lookback_t-1-k)*step_ns)) for k in range(lookback_t)]

# =========================
# DATASET
# =========================
class FixedKNNSubgraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_by_tid, mu, sd, train_tids,
                 k_neighbors=K_NEIGHBORS, max_windows=MAX_TRAIN_WINDOWS_PER_TURBINE,
                 allow_impute=False):
        super().__init__()
        self.data_by_tid = data_by_tid
        self.mu, self.sd = mu, sd
        self.train_tids  = sorted(train_tids)
        self.allow_impute = allow_impute
        self.step_ns = np.int64(pd.Timedelta(minutes=FREQ_MIN).value)

        self.arrX, self.arry, self.tarr, self.tmap = {}, {}, {}, {}
        for tid, df in data_by_tid.items():
            tvals = df[TIME_COL].astype("int64").values.astype(np.int64)
            self.tarr[tid] = tvals
            self.tmap[tid] = {np.int64(t): i for i, t in enumerate(tvals)}
            self.arrX[tid] = standardize(df, mu, sd)
            self.arry[tid] = df[TARGET_COL].values.astype(np.float32)

        self.node_list, self.edge_index, self.edge_attr = {}, {}, {}
        for tid in self.train_tids:
            nodes = knn_ids_within_set(tid, self.train_tids, k_neighbors)
            self.node_list[tid] = nodes
            ei, ea = build_edges_for_node_list(nodes)
            self.edge_index[tid] = ei
            self.edge_attr[tid]  = ea

        self.samples = []
        for tid in self.train_tids:
            times = self.tarr[tid]
            nodes = self.node_list[tid]
            candidates = []
            for idx in range(LOOKBACK_T-1, len(times)):
                t = np.int64(times[idx])
                if allow_impute:
                    candidates.append(t)
                    continue
                lag_times = [np.int64(t-(LOOKBACK_T-1-k)*self.step_ns) for k in range(LOOKBACK_T)]
                ok = all(self.tmap[u].get(t) is not None and
                         all(self.tmap[u].get(tt) is not None for tt in lag_times)
                         for u in nodes)
                if ok:
                    candidates.append(t)
            if max_windows and len(candidates) > max_windows:
                sel = np.linspace(0, len(candidates)-1, max_windows).astype(int)
                candidates = [candidates[s] for s in sel]
            self.samples.extend([(tid, np.int64(t)) for t in candidates])
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        focus_tid, center_t = self.samples[idx]
        nodes = self.node_list[focus_tid]
        N, Fdim = len(nodes), len(FEAT_COLS)
        node_seq = np.zeros((N, LOOKBACK_T, Fdim), dtype=np.float32)
        y    = np.zeros((N,), dtype=np.float32)
        mask = np.zeros((N,), dtype=np.float32)
        mask[0] = 1.0

        for i_node, tid in enumerate(nodes):
            if self.allow_impute:
                idxs = get_lag_indices_impute(self.tarr[tid], center_t, self.step_ns, LOOKBACK_T)
                cidx = _nearest_index(self.tarr[tid], center_t)
            else:
                lag_times = [np.int64(center_t-(LOOKBACK_T-1-k)*self.step_ns) for k in range(LOOKBACK_T)]
                idxs = [self.tmap[tid][tt] for tt in lag_times]
                cidx = self.tmap[tid][center_t]
            node_seq[i_node] = self.arrX[tid][idxs,:]
            y[i_node]        = self.arry[tid][cidx]

        return Data(
            node_seq=torch.from_numpy(node_seq),
            y=torch.from_numpy(y),
            mask=torch.from_numpy(mask),
            edge_index=self.edge_index[focus_tid],
            edge_attr=self.edge_attr[focus_tid],
            focus_index=torch.tensor([0], dtype=torch.long),
        )

# =========================
# MODEL
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class EdgeAwareMP(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.0):
        super().__init__(aggr="mean")
        self.msg = nn.Sequential(
            nn.Linear(node_dim+edge_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(node_dim+hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
    def forward(self, x, edge_index, e): return self.propagate(edge_index=edge_index, x=x, e=e)
    def message(self, x_j, e): return self.msg(torch.cat([x_j, e], dim=-1))
    def update(self, aggr_out, x): return self.upd(torch.cat([x, aggr_out], dim=-1))

class STGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim):
        super().__init__()
        self.lstm = nn.LSTM(node_feat_dim, HIDDEN_CHANNELS, NUM_LSTM_LAYERS,
                            batch_first=True,
                            dropout=ENC_DROPOUT if NUM_LSTM_LAYERS > 1 else 0.0)
        self.node_enc = nn.Sequential(
            nn.Dropout(ENC_DROPOUT), nn.Linear(HIDDEN_CHANNELS, ENCODING_CHANNELS),
            nn.ReLU(), nn.Linear(ENCODING_CHANNELS, HIDDEN_CHANNELS), nn.ReLU(),
        )
        self.edge_enc = MLP(edge_feat_dim, ENCODING_CHANNELS, ENCODING_CHANNELS, ENC_DROPOUT)
        self.mp_layers = nn.ModuleList([
            EdgeAwareMP(HIDDEN_CHANNELS, ENCODING_CHANNELS, HIDDEN_CHANNELS, GNN_DROPOUT)
            for _ in range(NUM_MP_LAYERS)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS), nn.ReLU(),
            nn.Dropout(GNN_DROPOUT), nn.Linear(HIDDEN_CHANNELS, 1),
        )
    def forward(self, node_seq, edge_index, edge_attr):
        out, _ = self.lstm(node_seq)
        h = self.node_enc(out[:,-1,:])
        e = self.edge_enc(edge_attr)
        for mp in self.mp_layers:
            h = h + mp(h, edge_index, e)
        return torch.sigmoid(self.decoder(h)).squeeze(-1)

# =========================
# TRAIN
# =========================
def train_loo_model(target_tid):
    train_tids = [t for t in TURBINE_IDS if t != target_tid]
    train_data = {}
    for t in train_tids:
        df = read_turbine_year(t, 2017)
        if df is None:
            raise RuntimeError(f"Missing Turbine{t}_2017.csv")
        train_data[t] = preprocess_df(df)

    mu, sd = compute_scaler(train_data)
    ds = FixedKNNSubgraphDataset(train_data, mu, sd, train_tids,
                                  allow_impute=TRAIN_ALLOW_IMPUTE)
    print(f"  train samples: {len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model = STGNN(len(FEAT_COLS), 3).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    use_amp = (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    t_fit = time.time()
    for epoch in range(1, EPOCHS+1):
        if epoch > 1:
            for g in opt.param_groups: g["lr"] *= LR_DECAY
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                yhat = model(batch.node_seq, batch.edge_index, batch.edge_attr)
                loss = F.mse_loss(yhat[batch.mask > 0.5], batch.y[batch.mask > 0.5])
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        print(f"  epoch {epoch:02d} mse={np.mean(losses):.6f}", flush=True)

    return model, mu, sd, time.time() - t_fit

# =========================
# EVAL
# =========================
@torch.no_grad()
def eval_target(model, mu, sd, target_tid, year):
    data_by_tid = {}
    for t in TURBINE_IDS:
        df = read_turbine_year(t, year)
        if df is not None:
            data_by_tid[t] = preprocess_df(df)
    if target_tid not in data_by_tid:
        return float("nan"), 0.0

    avail = sorted(data_by_tid.keys())
    nodes = knn_ids_within_set(target_tid, avail, K_NEIGHBORS)
    step_ns = np.int64(pd.Timedelta(minutes=FREQ_MIN).value)

    arrX, arry, tarr = {}, {}, {}
    for tid in nodes:
        df = data_by_tid[tid]
        tvals = df[TIME_COL].astype("int64").values.astype(np.int64)
        tarr[tid] = tvals
        arrX[tid] = standardize(df, mu, sd)
        arry[tid] = df[TARGET_COL].values.astype(np.float32)

    edge_index, edge_attr = build_edges_for_node_list(nodes)
    edge_index = edge_index.to(DEVICE)
    edge_attr  = edge_attr.to(DEVICE)

    focus_times = tarr[target_tid]
    model.eval()
    t0 = time.time()
    preds = []
    trues = []

    for idx in range(LOOKBACK_T-1, len(focus_times)):
        center_t = np.int64(focus_times[idx])
        N = len(nodes)
        node_seq = np.zeros((N, LOOKBACK_T, len(FEAT_COLS)), dtype=np.float32)
        for i_node, tid in enumerate(nodes):
            idxs = get_lag_indices_impute(tarr[tid], center_t, step_ns, LOOKBACK_T)
            node_seq[i_node] = arrX[tid][idxs,:]
        node_seq_t = torch.from_numpy(node_seq).to(DEVICE, non_blocking=True)
        yhat = model(node_seq_t, edge_index, edge_attr)
        preds.append(float(yhat[0].cpu().item()) * 100.0)
        trues.append(float(arry[target_tid][idx]) * 100.0)

    pred_time = time.time() - t0
    preds = np.array(preds)
    trues = np.array(trues)
    rmse  = float(np.sqrt(np.mean((preds - trues)**2)))
    return rmse, pred_time

# =========================
# RESUME HELPERS
# =========================
def load_done_keys():
    done = set()
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        try:
            prev = pd.read_csv(DETAIL_FILE)
            if {"target","year"}.issubset(prev.columns):
                done = set(f"{int(r.target)}|{int(r.year)}" for _, r in prev.iterrows())
                print(f"[INFO] Resuming — {len(done)} pairs already done.", flush=True)
        except Exception:
            pass
    return done

def append_row(row):
    row_df = pd.DataFrame([row])
    write_header = not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0
    row_df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)

def write_summary():
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        detail_df  = pd.read_csv(DETAIL_FILE)
        summary_df = detail_df.groupby(["method","year"], as_index=False).agg(
            avg_rmse=("rmse","mean"),
            total_runtime_sec=("runtime_sec","sum"),
        )
        summary_df.to_csv(SUMMARY_FILE, index=False)
        print("[DONE] Summary saved.", flush=True)

# =========================
# MAIN
# =========================
def main():
    print(f"DEVICE: {DEVICE} | TEST_YEARS: {TEST_YEARS}", flush=True)
    done_keys = load_done_keys()

    for target_tid in TURBINE_IDS:
        years_needed = [y for y in TEST_YEARS
                        if f"{target_tid}|{y}" not in done_keys
                        and not (y == 2018 and target_tid in EXCLUDE_2018)]
        if not years_needed:
            print(f"[SKIP] Turbine {target_tid} — all years done.", flush=True)
            continue

        print(f"\n=== LOO Target Turbine {target_tid} ===", flush=True)
        try:
            model, mu, sd, fit_time = train_loo_model(target_tid)
        except Exception as e:
            print(f"  [ERROR] train: {e}", flush=True)
            continue

        print(f"  fit_time={fit_time:.1f}s", flush=True)

        for year in years_needed:
            print(f"  Evaluating year {year} ...", flush=True)
            try:
                rmse_val, pred_time = eval_target(model, mu, sd, target_tid, year)
            except Exception as e:
                print(f"  [ERROR] eval year {year}: {e}", flush=True)
                continue

            append_row({
                "method":      "P_GNN_LOO",
                "target":      target_tid,
                "year":        year,
                "rmse":        rmse_val,
                "runtime_sec": fit_time + pred_time,
                "fit_time_sec": fit_time,
                "pred_time_sec": pred_time,
            })
            done_keys.add(f"{target_tid}|{year}")
            print(f"  -> Saved year {year}. RMSE={rmse_val:.4f}", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary()


if __name__ == "__main__":
    main()
