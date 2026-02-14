# -*- coding: utf-8 -*-
"""
LG Aimers — 7일 메뉴 수요예측 (CPU)
A) XGB/LGB + (옵션) CNN-LSTM + 요일보정 + ZI + 캡
B) FT-Dozer(범주 임베딩 + Dozer Attention, PyTorch/CPU)
=> 검증 기반 late-fusion 가중치 자동 탐색 → 테스트 7일 롤아웃에 적용

- 전부 CPU 동작
- 달력 플래그 미지 값은 주말만 요일로 보정
- 최종 예측값은 최소 1 보장(0 없음)
"""

import os, glob, warnings, random, math
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)

# ===== 경로 설정 (Windows) =====
BASE_DIR   = Path(r"C:\Users\minseo\lg")
TRAIN_FILE = BASE_DIR / "train_07" / "re_train_17.csv"        # ✅ 이 파일만 사용
TEST_GLOB  = str(BASE_DIR / "test_08" / "TEST_*_processed.csv")
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
OUT_PATH   = BASE_DIR / "submission_fused.csv"

# ===== 하이퍼파라미터 =====
VALID_RATIO     = 0.12
EARLY_STOP_XGB  = 200
EARLY_STOP_LGB  = 300
NUM_ROUNDS_XGB  = 5000
NUM_ROUNDS_LGB  = 8000

DL_WIN          = 35
DL_DROPOUT      = 0.2

# ZI 라우팅
ZI_ZERO_RATIO   = 0.65
ZI_CV           = 1.2
ZI_N_POS        = 6
ALPHA_MAX       = 0.45

# ===== 유틸 =====
def safe_read_csv(path):
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 로드 실패: {path}")

def ensure_basic_cols(df):
    for c in ["영업일자","영업장명_메뉴명","매출수량"]:
        if c not in df.columns: raise ValueError(f"필수 컬럼 누락: {c}")
    return df

def normalize_series_from_group(g):
    g = g.copy()
    g["영업일자"] = pd.to_datetime(g["영업일자"]).dt.normalize()
    s = g.groupby("영업일자")["매출수량"].sum().astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def pull_val(series, d):
    v = series.get(d, 0.0)
    if isinstance(v, pd.Series): v = v.sum()
    return float(v)

def list_test_files():
    files = sorted(glob.glob(TEST_GLOB))
    if not files:
        raise FileNotFoundError("TEST_*_processed.csv 경로를 찾을 수 없습니다.")
    return files

# ===== 달력 플래그 처리 =====
CAL_FLAG_COLS = [
    "is_holiday","is_before_holiday","is_after_holiday",
    "is_sandwich","between_holidays","is_weekend",
]

def build_calendar_flag_map(df):
    dfx = df.copy()
    dfx["영업일자"] = pd.to_datetime(dfx["영업일자"]).dt.normalize()
    for c in CAL_FLAG_COLS:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce").fillna(0.0).clip(0, 1)
    g = dfx.groupby("영업일자")
    cal_map = {}
    for dt, grp in g:
        flags = {}
        for c in CAL_FLAG_COLS:
            if c in grp.columns:
                v = float(np.rint(np.nanmean(pd.to_numeric(grp[c], errors="coerce").fillna(0).values)))
                flags[c] = int(max(0, min(1, v)))
        cal_map[pd.to_datetime(dt)] = flags
    return cal_map

def get_flags_for_date(cal_map, target_date):
    flags = {c:0 for c in CAL_FLAG_COLS}
    if target_date in cal_map:
        for k,v in cal_map[target_date].items():
            flags[k] = int(v)
    if flags.get("is_weekend", 0) == 0 and target_date.weekday() in (5, 6):
        flags["is_weekend"] = 1
    return flags

# ===== 요일-주차 약한 보정(최근 8주) =====
def weekday_multiplier(series, target_date, weeks=8):
    end = target_date - timedelta(days=1)
    start = end - timedelta(days=7*weeks - 1)
    hist = series.loc[(series.index >= start) & (series.index <= end)]
    if hist.empty: return 1.0
    same_w = hist[hist.index.weekday == target_date.weekday()]
    denom = hist.mean() if hist.size > 0 else 0.0
    numer = same_w.mean() if same_w.size > 0 else 0.0
    if denom is None or denom <= 1e-9: return 1.0
    ratio = float(numer/denom)
    return float(np.clip(ratio, 0.85, 1.15))

# ===== 피처 생성 (A 파이프라인) =====
def build_feats_from_series(values, target_date, cal_map):
    lag = lambda n: pull_val(values, target_date - timedelta(days=n))
    def roll(win):
        arr = np.array([pull_val(values, target_date - timedelta(days=n)) for n in range(1, win+1)], float)
        if arr.size == 0: return 0,0,0,0
        return arr.mean(), arr.std(ddof=0), arr.max(), arr.min()

    l1,l7,l14,l28 = [lag(n) for n in (1,7,14,28)]
    rm7,rs7,rmax7,rmin7   = roll(7)
    rm14,rs14,rmax14,rmin14 = roll(14)
    rm28,rs28,rmax28,rmin28 = roll(28)
    arr28 = [pull_val(values, target_date - timedelta(days=n)) for n in range(1,29)]
    nz28  = float(np.mean(np.array(arr28)>0)) if arr28 else 0.0
    dsl=365.0
    for n,v in enumerate(arr28,1):
        if v>0: dsl=float(n); break

    flags = get_flags_for_date(cal_map, target_date)

    dow = target_date.weekday()
    month = target_date.month
    dow_sin, dow_cos = np.sin(2*np.pi*dow/7.0), np.cos(2*np.pi*dow/7.0)
    month_sin, month_cos = np.sin(2*np.pi*month/12.0), np.cos(2*np.pi*month/12.0)

    rm7_over_rm28 = float(rm7 / (rm28 + 1e-6)) if rm28 > 0 else 1.0

    out = {
        "dow": dow, "month": month,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "lag1": l1, "lag7": l7, "lag14": l14, "lag28": l28,
        "rm7": rm7, "rs7": rs7, "rmax7": rmax7, "rmin7": rmin7,
        "rm14": rm14, "rs14": rs14, "rmax14": rmax14, "rmin14": rmin14,
        "rm28": rm28, "rs28": rs28, "rmax28": rmax28, "rmin28": rmin28,
        "rm7_over_rm28": rm7_over_rm28,
        "nz28": nz28, "dsl": dsl,
        "is_holiday": flags.get("is_holiday", 0),
        "is_before_holiday": flags.get("is_before_holiday", 0),
        "is_after_holiday": flags.get("is_after_holiday", 0),
        "is_sandwich": flags.get("is_sandwich", 0),
        "between_holidays": flags.get("between_holidays", 0),
        "is_weekend": flags.get("is_weekend", 0),
    }
    return out

FEATS = [
    "dow","month","dow_sin","dow_cos","month_sin","month_cos",
    "lag1","lag7","lag14","lag28",
    "rm7","rs7","rmax7","rmin7","rm14","rs14","rmax14","rmin14",
    "rm28","rs28","rmax28","rmin28","rm7_over_rm28",
    "nz28","dsl",
    "is_holiday","is_before_holiday","is_after_holiday",
    "is_sandwich","between_holidays","is_weekend",
]

# ===== 라우팅/ZI =====
def routing_stats_from_last28(series,last_date):
    idx=[last_date-timedelta(days=d) for d in range(27,-1,-1)]
    vals=np.array([pull_val(series,d) for d in idx],float)
    zero_ratio=float((vals==0).mean()) if vals.size else 1.0
    pos=vals[vals>0]; n_pos=int(pos.size)
    pos_mean=pos.mean() if n_pos else 0.0
    pos_std =pos.std(ddof=0) if n_pos else 0.0
    cv=pos_std/pos_mean if pos_mean>0 else np.inf
    var=pos.var(ddof=0) if n_pos else 0.0
    phi=var/pos_mean if pos_mean>1e-9 else np.inf
    p95=np.percentile(pos,95) if n_pos else 0.0
    return dict(zero_ratio=zero_ratio,n_pos=n_pos,cv=cv,phi=phi,
                p_nonzero=1-zero_ratio,pos_mean=pos_mean,pos_p95=p95)

def zi_surrogate_pred(stats,model_type):
    base=stats["p_nonzero"]*stats["pos_mean"]
    if model_type=="ZINB":
        shrink=1.0/(1.0+max(0.0,stats["phi"]-1.0))
        return max(0.0, base*(0.7+0.3*shrink))
    return max(0.0,base)

# ===== 라이브러리 로드 =====
xgb_ok=True
try:
    import xgboost as xgb
except Exception:
    xgb_ok=False

lgb_ok=True
try:
    import lightgbm as lgb
except Exception:
    lgb_ok=False

dl_ok=True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.config.set_visible_devices([], 'GPU')
    tf.random.set_seed(42)
except Exception:
    dl_ok=False

from sklearn.ensemble import GradientBoostingRegressor

# ===== 학습 데이터(A용) =====
def build_supervised_rows(train, cal_map, min_past=DL_WIN):
    rows,ys=[],[]
    meta=[]
    for item,g in train.groupby("영업장명_메뉴명"):
        s=normalize_series_from_group(g)
        for i,dt in enumerate(s.index):
            past=s.iloc[:i]
            if past.size<min_past: continue
            y=float(s.iloc[i])
            if y<=0: continue
            rows.append(build_feats_from_series(past,dt,cal_map)); ys.append(y)
            meta.append((item, dt))
    X=pd.DataFrame(rows,columns=FEATS); y=np.array(ys,float)
    if X.empty: raise RuntimeError("학습 샘플 없음")
    return X,y,meta

def build_sequence_dataset_only(train, win=DL_WIN):
    X_seq, y_seq = [], []
    for _,g in train.groupby("영업장명_메뉴명"):
        s = normalize_series_from_group(g).astype(float)
        vals = s.values
        if len(vals) <= win: continue
        for i in range(win, len(vals)):
            X_seq.append(np.log1p(vals[i-win:i]))
            y_seq.append(np.log1p(max(vals[i], 0.0)))
    if not X_seq: raise RuntimeError("시퀀스 학습 샘플 없음")
    X_seq = np.array(X_seq, dtype=np.float32)[..., None]
    y_seq = np.array(y_seq, dtype=np.float32)
    return X_seq, y_seq

def time_order_split_idx(n, valid_ratio=0.1):
    n_val = max(1, int(n * valid_ratio))
    idx_train = np.arange(0, n - n_val)
    idx_val   = np.arange(n - n_val, n)
    return idx_train, idx_val

# ===== sMAPE =====
def _smape(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)
    y_pred = np.clip(y_pred, 1e-9, None)
    return (200.0/len(y_true)) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# ===== XGBoost (CPU) =====
def train_xgb_model(train, cal_map):
    X, y, _ = build_supervised_rows(train, cal_map, min_past=DL_WIN)
    idx_tr, idx_val = time_order_split_idx(len(X), VALID_RATIO)
    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_val, y_val = X.iloc[idx_val], y[idx_val]

    if xgb_ok:
        models = []
        for sd in (42,202,777):
            dtr  = xgb.DMatrix(X_tr,  label=np.log1p(y_tr))
            dval = xgb.DMatrix(X_val, label=np.log1p(y_val))
            params=dict(objective="reg:squarederror", eval_metric="mae",
                        tree_method="hist", predictor="cpu_predictor",
                        max_bin=256, eta=0.03, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=8, reg_lambda=1.0, seed=sd)
            model = xgb.train(params, dtr, num_boost_round=NUM_ROUNDS_XGB,
                              evals=[(dtr,"train"),(dval,"valid")],
                              early_stopping_rounds=EARLY_STOP_XGB, verbose_eval=False)
            models.append(model)

        def _predict_with_best(model, dmatrix):
            bi = getattr(model, "best_iteration", None)
            if isinstance(bi, (int, np.integer)) and bi >= 0:
                return model.predict(dmatrix, iteration_range=(0, int(bi) + 1))
            bntl = getattr(model, "best_ntree_limit", None)
            if isinstance(bntl, (int, np.integer)) and bntl > 0:
                try: return model.predict(dmatrix, iteration_range=(0, int(bntl)))
                except Exception: pass
            try:
                rounds = model.num_boosted_rounds()
                return model.predict(dmatrix, iteration_range=(0, int(rounds)))
            except Exception:
                return model.predict(dmatrix)

        def predict(Xdf):
            d = xgb.DMatrix(Xdf)
            ps = [_predict_with_best(m, d) for m in models]
            ps = [np.expm1(np.clip(p, 0, None)) for p in ps]
            return np.mean(ps, axis=0)
    else:
        gbr=GradientBoostingRegressor(learning_rate=0.05,n_estimators=1000,max_depth=6,subsample=0.9)
        gbr.fit(X_tr, np.log1p(y_tr))
        def predict(Xdf): return np.expm1(np.clip(gbr.predict(Xdf),0,None))
    return predict, (X, y)

# ===== LightGBM (CPU) =====
def train_lgb_model(train, cal_map):
    X, y, _ = build_supervised_rows(train, cal_map, min_past=DL_WIN)
    idx_tr, idx_val = time_order_split_idx(len(X), VALID_RATIO)
    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_val, y_val = X.iloc[idx_val], y[idx_val]

    if lgb_ok:
        models = []
        for sd in (13,101):
            dtr  = lgb.Dataset(X_tr,  label=np.log1p(y_tr), free_raw_data=False)
            dval = lgb.Dataset(X_val, label=np.log1p(y_val), free_raw_data=False)
            params=dict(objective="regression", metric="l1", learning_rate=0.05,
                        num_leaves=31, feature_fraction=0.88, bagging_fraction=0.80,
                        bagging_freq=1, min_data_in_leaf=40, lambda_l2=2.0,
                        max_bin=255, device="cpu", seed=sd)
            m=lgb.train(params, dtr, num_boost_round=NUM_ROUNDS_LGB,
                        valid_sets=[dtr,dval], valid_names=["train","valid"],
                        callbacks=[lgb.early_stopping(EARLY_STOP_LGB, verbose=False),
                                   lgb.log_evaluation(period=0)])
            models.append(m)

        def predict(Xdf):
            ps=[m.predict(Xdf, num_iteration=m.best_iteration) for m in models]
            p=np.mean(ps, axis=0)
            return np.expm1(np.clip(p, 0, None))
    else:
        def predict(Xdf): return np.zeros(len(Xdf), dtype=float)
    return predict

# ===== CNN-LSTM (CPU, 없으면 스킵) =====
def train_cnn_lstm_model(train, win=DL_WIN, seed=1234):
    if not dl_ok:
        def predict(seq_): return np.zeros(len(seq_), dtype=float)
        return predict
    try: tf.random.set_seed(seed)
    except Exception: pass

    X_seq, y_seq = build_sequence_dataset_only(train, win=win)
    n = len(X_seq); n_val = max(1, int(n * VALID_RATIO))
    X_tr, y_tr = X_seq[:n - n_val], y_seq[:n - n_val]
    X_val, y_val = X_seq[n - n_val:], y_seq[n - n_val:]

    ds_tr  = tf.data.Dataset.from_tensor_slices((X_tr,  y_tr)).batch(512).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(512).prefetch(tf.data.AUTOTUNE)

    inputs = keras.Input(shape=(win,1))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inputs)
    x = layers.Dropout(DL_DROPOUT)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DL_DROPOUT)(x)
    outputs = layers.Dense(1, dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mae")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]
    model.fit(ds_tr, epochs=50, validation_data=ds_val, verbose=0, callbacks=callbacks)

    def predict(seq_batch):
        if not seq_batch: return np.array([], dtype=float)
        Xb = np.array([np.log1p(np.array(s, dtype=np.float32)) for s in seq_batch])[..., None]
        pred_log = model.predict(Xb, verbose=0).reshape(-1)
        return np.expm1(np.maximum(pred_log, 0.0))
    return predict

# ===== FT-Dozer (B 파이프라인; 범주 임베딩) =====
CAT_COLS = [
    "weekday","is_holiday","is_weekend","is_sandwich",
    "is_before_holiday","is_after_holiday",
    "menu_category","quarter","season","solar_term"
]

def month_to_season_id(m):
    return {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}.get(int(m),0)

def base_from_dates(dates: pd.DatetimeIndex):
    dow = dates.weekday.values
    qtr = ((dates.month.values - 1)//3)
    sea = np.array([month_to_season_id(m) for m in dates.month.values], dtype=np.int64)
    wknd= (dow>=5).astype(np.int64)
    return {
        "weekday": dow.astype(np.int64),
        "is_weekend": wknd,
        "quarter": qtr.astype(np.int64),
        "season": sea,
        "is_holiday": np.zeros(len(dates), np.int64),
        "is_sandwich": np.zeros(len(dates), np.int64),
        "is_before_holiday": np.zeros(len(dates), np.int64),
        "is_after_holiday": np.zeros(len(dates), np.int64),
        "solar_term": np.zeros(len(dates), np.int64),
    }

torch_ok=True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try: torch.set_num_threads(max(1, os.cpu_count()//2))
    except Exception: pass
except Exception:
    torch_ok=False

def train_ft_dozer(train):
    if not torch_ok:
        def predict_seq(seq_win, cats_win, menu_id): return np.zeros(len(seq_win), dtype=float)
        return predict_seq, None, None, 1

    menus = sorted(train["영업장명_메뉴명"].unique().tolist())
    menu2id = {m:i for i,m in enumerate(menus)}
    if "menu_category" not in train.columns:
        train["menu_category"]=0
    menu_cat_mode = (
        train.groupby("영업장명_메뉴명")["menu_category"]
             .agg(lambda s: int(pd.to_numeric(s, errors="coerce").dropna().astype(int).value_counts().index[0]) if len(pd.to_numeric(s, errors="coerce").dropna()) else 0)
             .to_dict()
    )

    cal_map_df = (train[["영업일자"] + [c for c in CAT_COLS if c in train.columns]]
                  .drop_duplicates("영업일자").set_index("영업일자").sort_index())

    # 카디널리티 추정
    def infer_cardinality():
        card={}
        for c in CAT_COLS:
            if c=="weekday": card[c]=7
            elif c in ["quarter","season"]: card[c]=4
            elif c in ["is_holiday","is_weekend","is_sandwich","is_before_holiday","is_after_holiday"]:
                card[c]=2
            elif c=="menu_category":
                vmax = int(pd.to_numeric(train["menu_category"], errors="coerce").fillna(0).max())
                card[c]=max(1, vmax+1)
            else:  # solar_term
                vmax = int(pd.to_numeric(train.get(c, pd.Series([0])), errors="coerce").fillna(0).max())
                card[c]=max(1, vmax+1)
        return card
    cat_card = infer_cardinality()

    def cats_for_dates_idx(dates: pd.DatetimeIndex, menu_name: str):
        base = base_from_dates(dates)
        re = cal_map_df.reindex(dates)
        cats={}
        for c in CAT_COLS:
            if c=="menu_category":
                val=int(menu_cat_mode.get(menu_name,0))
                cats[c]=np.full(len(dates), val, dtype=np.int64)
                continue
            if c in re.columns:
                arr = pd.to_numeric(re[c], errors="coerce").to_numpy(dtype=float)
                fb  = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
                filled = np.where(np.isnan(arr), fb, arr)
            else:
                filled = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
            filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
            # 안전 클립
            maxv = int(cat_card[c]-1)
            filled = np.clip(filled, 0, maxv if maxv>=0 else 0).astype(np.int64)
            cats[c]=filled
        return cats

    # 학습 샘플 구성
    Xc_list, y_list, mid_list = [], [], []
    Cat_lists = {k: [] for k in CAT_COLS}
    for item, g in train.groupby("영업장명_메뉴명"):
        s = normalize_series_from_group(g)
        if s.empty: continue
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        y_full = s.reindex(full_idx).fillna(0.0).astype(float)
        cats_all = cats_for_dates_idx(full_idx, item)
        for i in range(DL_WIN, len(full_idx)):
            y = float(y_full.iloc[i])
            if y<=0: continue
            prev = y_full.iloc[i-DL_WIN:i].values
            Xc_list.append(np.log1p(prev)[:,None].astype(np.float32))
            y_list.append(y)
            mid_list.append(menu2id[item])
            for k in CAT_COLS:
                Cat_lists[k].append(cats_all[k][i-DL_WIN:i])

    if not Xc_list:
        def predict_seq(seq_win, cats_win, menu_id): return np.zeros(len(seq_win), dtype=float)
        return predict_seq, None, None, 1

    Xc = np.stack(Xc_list, axis=0)       # [N,WIN,1]
    yb = np.array(y_list, np.float32)    # [N]
    mid= np.array(mid_list, np.int64)    # [N]
    Cats = {k: np.stack(v, axis=0).astype(np.int64) for k,v in Cat_lists.items()}  # [N,WIN]

    # Torch Dataset
    class DS(torch.utils.data.Dataset):
        def __init__(self, Xc,yb,mid,Cats):
            self.Xc=torch.from_numpy(Xc); self.y=torch.from_numpy(yb)
            self.mid=torch.from_numpy(mid)
            self.cats={k: torch.from_numpy(v) for k,v in Cats.items()}
            self.keys=list(self.cats.keys())
        def __len__(self): return len(self.y)
        def __getitem__(self,i):
            return self.Xc[i], self.y[i], self.mid[i], {k:self.cats[k][i] for k in self.keys}

    D_MODEL=64; N_HEAD=4; N_BLOCKS=2; DROPOUT=0.1

    class DozerAttention(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1):
            super().__init__()
            self.d_model=d_model; self.nhead=nhead; self.dk=d_model//nhead
            self.q_proj=nn.Linear(d_model, d_model); self.k_proj=nn.Linear(d_model, d_model)
            self.v_proj=nn.Linear(d_model, d_model); self.o_proj=nn.Linear(d_model, d_model)
            self.out_drop=nn.Dropout(dropout)
            self.lambda_decay=nn.Parameter(torch.tensor(0.03))
            self.flag_gate=nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        def forward(self, x, flags=None):
            B,T,D=x.shape
            q=self.q_proj(x).view(B,T,self.nhead,self.dk).transpose(1,2)
            k=self.k_proj(x).view(B,T,self.nhead,self.dk).transpose(1,2)
            v=self.v_proj(x).view(B,T,self.nhead,self.dk).transpose(1,2)
            # ----- flag gating -----
            gate_in = flags if flags is not None else x           # [B,T,D]
            g = self.flag_gate(gate_in)                            # [B,T,D]
            g = g.view(B,T,self.nhead,self.dk).transpose(1,2)     # [B,heads,T,dk]
            v = v * g
            # ----- distance decay attention -----
            attn=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.dk)
            with torch.no_grad():
                dist=torch.arange(T,device=x.device).unsqueeze(0)-torch.arange(T,device=x.device).unsqueeze(1)
                dist=dist.abs().float()
            lam=F.softplus(self.lambda_decay)
            attn=torch.softmax(attn - lam*dist, dim=-1)
            out=torch.matmul(attn,v).transpose(1,2).contiguous().view(B,T,D)
            return self.out_drop(self.o_proj(out))

    class Block(nn.Module):
        def __init__(self,d_model,nhead,dropout=0.1,mlp_ratio=2.0):
            super().__init__()
            self.attn=DozerAttention(d_model,nhead,dropout)
            self.ln1=nn.LayerNorm(d_model)
            self.mlp=nn.Sequential(nn.Linear(d_model,int(d_model*mlp_ratio)), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(int(d_model*mlp_ratio), d_model),
                                   nn.Dropout(dropout))
            self.ln2=nn.LayerNorm(d_model)
        def forward(self,x,flags=None):
            x=x+self.attn(self.ln1(x),flags=flags)
            x=x+self.mlp(self.ln2(x))
            return x

    def emb_dim(card): return int(min(16, max(4, round((card**0.5)*4))))
    class FTDozer(nn.Module):
        def __init__(self, n_menu, cat_card):
            super().__init__()
            self.menu_emb=nn.Embedding(n_menu, D_MODEL)
            self.cont_proj=nn.Linear(1, D_MODEL)
            self.pos_emb=nn.Parameter(torch.randn(1, DL_WIN, D_MODEL)*0.02)
            self.cat_embs=nn.ModuleDict()
            for c,card in cat_card.items():
                self.cat_embs[c]=nn.Embedding(int(card), emb_dim(int(card)))
            self.in_proj=nn.Linear(D_MODEL + sum(emb_dim(int(card)) for card in cat_card.values()), D_MODEL)
            # NEW: flag(5) → D 모델 투영기 (플래그 = 공휴일/주말/샌드위치/전날/다음날)
            self.flag_proj = nn.Sequential(
                nn.Linear(5, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, D_MODEL)
            )
            self.blocks=nn.ModuleList([Block(D_MODEL,N_HEAD,DROPOUT) for _ in range(N_BLOCKS)])
            self.head=nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))
        def forward(self, x_cont, cats_dict, menu_id):
            # x_cont: [B,T,1], cats: each [B,T]
            B,T,_=x_cont.shape
            cont=self.cont_proj(x_cont)                    # [B,T,D]
            cat_es=[self.cat_embs[k](cats_dict[k]) for k in CAT_COLS if k in self.cat_embs]
            cat_cat=torch.cat(cat_es, dim=-1)              # [B,T,SumE]
            h=torch.cat([cont, cat_cat], dim=-1)
            h=self.in_proj(h)+self.pos_emb[:, :T, :]
            # flag summary → D 차원으로 투영
            flag_keys = ["is_holiday","is_weekend","is_sandwich","is_before_holiday","is_after_holiday"]
            if all(k in cats_dict for k in flag_keys):
                flags5 = torch.stack([cats_dict[k] for k in flag_keys], dim=-1).float()  # [B,T,5]
                flagsD = self.flag_proj(flags5)  # [B,T,D]
            else:
                flagsD = None
            for blk in self.blocks:
                h=blk(h, flags=flagsD)
            # 메뉴 bias
            mvec=self.menu_emb(menu_id)[:,None,:].expand(B,T,-1)
            h=h+mvec
            out=self.head(h[:,-1,:]).squeeze(-1)          # [B]
            out=F.softplus(out)                            # >=0
            return out

    # 데이터/로더
    class DS(torch.utils.data.Dataset):
        def __init__(self, Xc,yb,mid,Cats):
            self.Xc=torch.from_numpy(Xc); self.y=torch.from_numpy(yb)
            self.mid=torch.from_numpy(mid)
            self.cats={k: torch.from_numpy(v) for k,v in Cats.items()}
            self.keys=list(self.cats.keys())
        def __len__(self): return len(self.y)
        def __getitem__(self,i):
            return self.Xc[i], self.y[i], self.mid[i], {k:self.cats[k][i] for k in self.keys}

    ds = DS(Xc, yb, mid, Cats)
    n = len(ds); n_val = max(1, int(n*VALID_RATIO))
    idx_tr = np.arange(0, n-n_val); idx_val=np.arange(n-n_val, n)
    dl_tr=torch.utils.data.DataLoader(ds, batch_size=256, sampler=torch.utils.data.SubsetRandomSampler(idx_tr.tolist()))
    dl_va=torch.utils.data.DataLoader(ds, batch_size=256, sampler=torch.utils.data.SequentialSampler(idx_val.tolist()))

    device="cpu"
    model=FTDozer(len(menus), cat_card).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=3e-3)
    best_loss=float("inf"); patience, bad=6, 0

    for ep in range(50):
        model.train(); tl=0.0; nbt=0
        for xc,y,midb,cats in dl_tr:
            xc=xc.to(device); y=y.to(device); midb=midb.to(device)
            cats={k:v.to(device) for k,v in cats.items()}
            opt.zero_grad()
            pred=model(xc, cats, midb)
            loss=F.l1_loss(pred, y)  # 안정적: 직접 MAE
            loss.backward(); opt.step()
            tl += float(loss.item()); nbt += 1
        model.eval(); vl=0.0; nvb=0
        with torch.no_grad():
            for xc,y,midb,cats in dl_va:
                xc=xc.to(device); y=y.to(device); midb=midb.to(device)
                cats={k:v.to(device) for k,v in cats.items()}
                pred=model(xc, cats, midb)
                loss=F.l1_loss(pred, y)
                vl += float(loss.item()); nvb += 1
        vl /= max(1, nvb)
        # print(f"[FT-Dozer] epoch {ep} val_mae {vl:.4f}")
        if vl < best_loss - 1e-4:
            best_loss=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad += 1
            if bad>=patience:
                print("[FT-Dozer] Early stopping."); break

    if 'best_state' in locals():
        model.load_state_dict(best_state)

    def predict_seq(seq_win_list, cats_win_list, menu_id_list):
        model.eval()
        outs=[]
        with torch.no_grad():
            Xc = torch.tensor(np.array([np.log1p(np.array(s, dtype=np.float32))[:,None] for s in seq_win_list]),
                              dtype=torch.float32)
            catsT = {}
            for k in CAT_COLS:
                arr = np.array([c[k] for c in cats_win_list], dtype=np.int64)
                catsT[k]=torch.tensor(arr, dtype=torch.long)
            mid = torch.tensor(menu_id_list, dtype=torch.long)
            pred = model(Xc, catsT, mid).cpu().numpy()
            pred = np.maximum(pred, 0.0)
        return pred

    return predict_seq, menus, menu2id, cat_card

# ===== A 파이프라인 모델 학습 =====
def train_A_models(train, train_cal_map):
    predict_xgb, (X_all, y_all) = train_xgb_model(train, train_cal_map)
    predict_lgb = train_lgb_model(train, train_cal_map)
    predict_dl  = train_cnn_lstm_model(train, win=DL_WIN)

    # 트리 내부 가중 탐색
    idx_tr, idx_val = time_order_split_idx(len(X_all), VALID_RATIO)
    X_val, y_val = X_all.iloc[idx_val], y_all[idx_val]
    p_x = predict_xgb(X_val) if xgb_ok else np.zeros(len(X_val))
    p_l = predict_lgb(X_val) if lgb_ok else np.zeros(len(X_val))
    cand_tree = [0.30, 0.45, 0.50, 0.55, 0.60]
    best_smape, best_w_tree = 1e9, 0.55
    for wt in cand_tree:
        if xgb_ok and lgb_ok:
            p_tree = wt * p_x + (1.0 - wt) * p_l
        else:
            p_tree = p_x if xgb_ok else p_l
        s = _smape(y_val, p_tree)
        if s < best_smape:
            best_smape, best_w_tree = s, wt
    W_TREE = float(best_w_tree)
    BASE_W_DL = 0.12
    print(f"[VAL] A-트리 가중치 W_TREE={W_TREE:.2f} (SMAPE={best_smape:.4f})")
    return predict_xgb, predict_lgb, predict_dl, W_TREE, BASE_W_DL

# ===== 융합 가중치 탐색(검증) =====
def find_fusion_weight(train, train_cal_map, pred_funcs_A, pred_B_bundle):
    predict_xgb, predict_lgb, predict_dl, W_TREE, BASE_W_DL = pred_funcs_A
    predict_B, menus, menu2id, cat_card = pred_B_bundle

    rows_A, rows_B, ys = [], [], []
    cal_map_df = (train[["영업일자"] + [c for c in CAT_COLS if c in train.columns]]
                  .drop_duplicates("영업일자").set_index("영업일자").sort_index())

    def cats_for_dates_idx(dates, menu_name):
        base = base_from_dates(dates)
        re = cal_map_df.reindex(dates)
        cats={}
        for c in CAT_COLS:
            if c=="menu_category":
                if "menu_category" in train.columns and not train.loc[train["영업장명_메뉴명"]==menu_name, "menu_category"].dropna().empty:
                    val=int(train.loc[train["영업장명_메뉴명"]==menu_name, "menu_category"].mode(dropna=True).iloc[0])
                else:
                    val=0
                cats[c]=np.full(len(dates), val, dtype=np.int64)
                continue
            if c in re.columns:
                arr = pd.to_numeric(re[c], errors="coerce").to_numpy(dtype=float)
                fb  = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
                filled = np.where(np.isnan(arr), fb, arr)
            else:
                filled = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
            filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
            cats[c]=filled.astype(np.int64)
        return cats

    for item, g in train.groupby("영업장명_메뉴명"):
        s = normalize_series_from_group(g)
        if s.empty: continue
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        y_full = s.reindex(full_idx).fillna(0.0).astype(float)
        cats_all = cats_for_dates_idx(full_idx, item)

        for i in range(DL_WIN, len(full_idx)):
            dt = full_idx[i]
            y = float(y_full.iloc[i])
            if y<=0: continue
            past_series = pd.Series(y_full.iloc[:i].values, index=full_idx[:i])
            flags = get_flags_for_date(train_cal_map, dt)
            # A
            feats = build_feats_from_series(past_series, dt, train_cal_map)
            Xdf = pd.DataFrame([feats], columns=FEATS).fillna(0)
            yx = float(predict_xgb(Xdf)[0]) if xgb_ok else 0.0
            yl = float(predict_lgb(Xdf)[0]) if lgb_ok else 0.0
            y_tree = (W_TREE*yx + (1.0-W_TREE)*yl) if (xgb_ok and lgb_ok) else (yx if xgb_ok else yl)
            wdl = BASE_W_DL
            if flags.get("is_holiday",0) or flags.get("is_sandwich",0) or flags.get("between_holidays",0): wdl += 0.05
            if flags.get("is_weekend",0): wdl += 0.03
            if flags.get("is_before_holiday",0) or flags.get("is_after_holiday",0): wdl += 0.02
            wdl = float(np.clip(wdl, 0.08, 0.22))
            seq = [pull_val(past_series, dt - timedelta(days=n)) for n in range(DL_WIN, 0, -1)]
            ydl = float(predict_dl([seq])[0]) if dl_ok else y_tree
            yA  = (1.0 - wdl) * y_tree + wdl * ydl
            yA *= weekday_multiplier(past_series, dt, weeks=8)
            rows_A.append(yA)
            # B
            if callable(predict_B) and menus is not None:
                cats_win = {k: cats_all[k][i-DL_WIN:i] for k in CAT_COLS}
                yB = float(predict_B([seq], [cats_win], [menu2id[item]])[0])
            else:
                yB = yA
            rows_B.append(yB)
            ys.append(y)

    if not rows_A:
        print("[FUSE] 정렬된 검증 샘플이 부족하여 A만 사용합니다.")
        return 1.0

    A = np.array(rows_A, float); B = np.array(rows_B, float); Y = np.array(ys, float)
    best_w, best_s = 1.0, _smape(Y, A)
    for w in np.linspace(0.0, 1.0, 11):
        pred = w*A + (1.0-w)*B
        s = _smape(Y, pred)
        if s < best_s:
            best_s, best_w = s, w
    print(f"[FUSE] Late-fusion weight w(A)={best_w:.2f} (sMAPE={best_s:.4f})")
    return float(best_w)

# ===== A 모델 학습 래퍼 =====
def train_A_models(train, train_cal_map):
    predict_xgb, (X_all, y_all) = train_xgb_model(train, train_cal_map)
    predict_lgb = train_lgb_model(train, train_cal_map)
    predict_dl  = train_cnn_lstm_model(train, win=DL_WIN)

    idx_tr, idx_val = time_order_split_idx(len(X_all), VALID_RATIO)
    X_val, y_val = X_all.iloc[idx_val], y_all[idx_val]
    p_x = predict_xgb(X_val) if xgb_ok else np.zeros(len(X_val))
    p_l = predict_lgb(X_val) if lgb_ok else np.zeros(len(X_val))
    cand_tree = [0.30, 0.45, 0.50, 0.55, 0.60]
    best_smape, best_w_tree = 1e9, 0.55
    for wt in cand_tree:
        p_tree = (wt * p_x + (1.0 - wt) * p_l) if (xgb_ok and lgb_ok) else (p_x if xgb_ok else p_l)
        s = _smape(y_val, p_tree)
        if s < best_smape:
            best_smape, best_w_tree = s, wt
    W_TREE = float(best_w_tree)
    BASE_W_DL = 0.12
    print(f"[VAL] A-트리 가중치 W_TREE={W_TREE:.2f} (SMAPE={best_smape:.4f})")
    return predict_xgb, predict_lgb, predict_dl, W_TREE, BASE_W_DL

# ===== 메인 =====
def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"훈련 데이터 없음: {TRAIN_FILE}")
    train=ensure_basic_cols(safe_read_csv(str(TRAIN_FILE)))
    train["영업일자"]=pd.to_datetime(train["영업일자"],errors="coerce")
    train["매출수량"]=pd.to_numeric(train["매출수량"],errors="coerce").fillna(0).clip(lower=0)

    # 캘린더 맵
    train_cal_map = build_calendar_flag_map(train)

    # (A) 모델 학습
    predict_xgb, predict_lgb, predict_dl, W_TREE, BASE_W_DL = train_A_models(train, train_cal_map)

    # (B) FT-Dozer 학습
    predict_B, menus, menu2id, cat_card = train_ft_dozer(train)

    # Late-fusion 가중 탐색
    wA = find_fusion_weight(train, train_cal_map,
                            (predict_xgb, predict_lgb, predict_dl, W_TREE, BASE_W_DL),
                            (predict_B, menus, menu2id, cat_card))

    # 예측
    all_rows=[]
    test_files = list_test_files()
    for tf in test_files:
        tdf=ensure_basic_cols(safe_read_csv(tf))
        tdf["영업일자"]=pd.to_datetime(tdf["영업일자"],errors="coerce")
        tdf["매출수량"]=pd.to_numeric(tdf["매출수량"],errors="coerce").fillna(0).clip(lower=0)
        test_cal_map = build_calendar_flag_map(tdf)
        merged_cal_map = dict(train_cal_map); merged_cal_map.update(test_cal_map)

        tag=os.path.basename(tf).split("_")[1]  # TEST_XX_processed.csv → XX
        test_tag=f"TEST_{tag}"
        last_date=tdf["영업일자"].max()

        # B용 도우미(날짜→범주)
        cal_map_df = (pd.concat([train[["영업일자"] + [c for c in CAT_COLS if c in train.columns]],
                                 tdf[["영업일자"] + [c for c in CAT_COLS if c in tdf.columns]]], axis=0)
                      .drop_duplicates("영업일자").set_index("영업일자").sort_index())

        def cats_for_dates_idx(dates, menu_name):
            base = base_from_dates(dates)
            re = cal_map_df.reindex(dates)
            cats={}
            for c in CAT_COLS:
                if c=="menu_category":
                    if "menu_category" in tdf.columns and not tdf.loc[tdf["영업장명_메뉴명"]==menu_name, "menu_category"].dropna().empty:
                        val=int(tdf.loc[tdf["영업장명_메뉴명"]==menu_name, "menu_category"].mode().iloc[0])
                    else:
                        val=0
                    cats[c]=np.full(len(dates), val, dtype=np.int64)
                    continue
                if c in re.columns:
                    arr = pd.to_numeric(re[c], errors="coerce").to_numpy(dtype=float)
                    fb  = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
                    filled = np.where(np.isnan(arr), fb, arr)
                else:
                    filled = base.get(c, np.zeros(len(dates), np.int64)).astype(float)
                filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
                cats[c]=filled.astype(np.int64)
            return cats

        for item,g in tdf.groupby("영업장명_메뉴명"):
            s=normalize_series_from_group(g)
            stats=routing_stats_from_last28(s,last_date)

            use_zi=(stats["zero_ratio"]>=ZI_ZERO_RATIO and stats["cv"]>=ZI_CV and stats["n_pos"]<=ZI_N_POS)
            if use_zi:
                zi_type="ZIP" if stats["phi"]<=1.3 else "ZINB"
                alpha=float(np.clip(0.1+0.15*max(0.0,stats["phi"]-1.3),0.2,ALPHA_MAX))
                zi_base=zi_surrogate_pred(stats,zi_type)
            else:
                alpha,zi_base=0.0,0.0

            def compute_cap(flags, stats):
                if stats["n_pos"]<=0: return 5.0
                if (flags.get("is_weekend",0) or flags.get("is_holiday",0)
                    or flags.get("is_sandwich",0) or flags.get("between_holidays",0)):
                    c = min(stats["pos_p95"]*2.3, stats["pos_mean"]*3.0)
                else:
                    c = min(stats["pos_p95"]*2.9, stats["pos_mean"]*3.6)
                return float(max(c, 3.0))

            cur=s.copy()
            # B용 메뉴 id
            menu_id = menu2id[item] if (menus is not None and item in menu2id) else 0

            for k in range(1,8):
                td=last_date+timedelta(days=k)
                flags = get_flags_for_date(merged_cal_map, td)
                feats=build_feats_from_series(cur,td,merged_cal_map)
                Xdf=pd.DataFrame([feats],columns=FEATS).fillna(0)

                # A: 트리 + DL 동적
                yx = float(predict_xgb(Xdf)[0]) if xgb_ok else 0.0
                yl = float(predict_lgb(Xdf)[0]) if lgb_ok else 0.0
                y_tree = (W_TREE*yx + (1.0-W_TREE)*yl) if (xgb_ok and lgb_ok) else (yx if xgb_ok else yl)

                wdl = BASE_W_DL
                if flags.get("is_holiday",0) or flags.get("is_sandwich",0) or flags.get("between_holidays",0): wdl += 0.05
                if flags.get("is_weekend",0): wdl += 0.03
                if flags.get("is_before_holiday",0) or flags.get("is_after_holiday",0): wdl += 0.02
                wdl = float(np.clip(wdl, 0.08, 0.22))

                seq = [pull_val(cur, td - timedelta(days=n)) for n in range(DL_WIN, 0, -1)]
                ydl = float(predict_dl([seq])[0]) if dl_ok else y_tree
                yA  = (1.0 - wdl) * y_tree + wdl * ydl
                yA *= weekday_multiplier(cur, td, weeks=8)

                # B: FT-Dozer 1-step
                if callable(predict_B) and menus is not None:
                    win_idx = pd.date_range(td - timedelta(days=DL_WIN), td - timedelta(days=1), freq="D")
                    cats_all = cats_for_dates_idx(win_idx, item)
                    cats_win = {k: cats_all[k] for k in CAT_COLS}
                    yB = float(predict_B([seq], [cats_win], [menu_id])[0])
                else:
                    yB = yA

                # Late fusion
                y_ens = wA * yA + (1.0 - wA) * yB

                # ZI + 캡 + 최소1
                cap = compute_cap(flags, stats)
                yhat = (1.0 - alpha)*y_ens + alpha*zi_base
                yhat = max(1.0, min(yhat, cap))

                all_rows.append((f"{test_tag}+{k}일",item,yhat))
                cur.loc[td]=yhat

    pred=pd.DataFrame(all_rows,columns=["영업일자","영업장명_메뉴명","pred"])
    wide=pred.pivot_table(index="영업일자",columns="영업장명_메뉴명",values="pred",aggfunc="sum").fillna(1.0)

    # 제출 형식
    if not SAMPLE_SUB.exists():
        idx = sorted(wide.index.tolist())
        cols = sorted(wide.columns.tolist())
        out = pd.DataFrame({"영업일자": idx})
        for c in cols: out[c]=wide[c].reindex(idx).fillna(1.0).values
    else:
        sample = safe_read_csv(str(SAMPLE_SUB))
        out = sample.copy()
        idx=out["영업일자"].tolist(); cols=out.columns[1:]
        vals=[]
        for lbl in idx:
            if lbl in wide.index:
                sr=wide.loc[lbl]
                vals.append([sr.get(c,1.0) for c in cols])
            else:
                vals.append([1.0]*len(cols))
        df_vals=pd.DataFrame(vals,columns=cols)
        out = pd.concat([pd.DataFrame({"영업일자": idx}), df_vals], axis=1)

    # 정수 반올림 + 최종 클립(>=1)
    for c in out.columns[1:]:
        out[c] = np.maximum(1, np.rint(out[c]).astype(int))

    # 저장
    save_path = str(OUT_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {save_path}")

if __name__=="__main__":
    main()
