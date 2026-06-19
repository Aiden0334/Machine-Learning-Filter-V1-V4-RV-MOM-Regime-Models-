"""
════════════════════════════════════════════════════════════════════════════════════════
  Model Performance Comparison between rule-based and ml-based — 2025-04-04 ~ 2026-06-10
════════════════════════════════════════════════════════════════════════════════════════
단독 OOS 검증 비교 
  학습: 2018-01-24 ~ 2025-04-03 (전체 IS)
  Test: 2025-04-04 ~ 2026-06-10 (진짜 forward, 약 14개월) 
  <첫 두달 FORWARD는 샘플 부족으로 1년치 늘렸음.>

프로세스 과정
  1. 전체 데이터 로드 + prep
  2. v1, v4 거래 추출 (전체)
  3. Forward 데이터 분리 (2026-04-04 이후)
  4. v1 ML 학습 (~2025-04-03 거래로) → Forward v1 거래 필터
  5. v4 ML 학습 (~2025-04-03 거래로) → Forward v4 거래 필터
  6. 결과 측정 (5 seed 평균)

비교
  베이스라인 (필터 X) vs ML 필터
  v1 단독, v4 단독, 50/50 합산

  → Key Findings

- XGBoost filtering did not improve V1 performance.
- XGBoost filtering improved V4 performance across multi-seeds. 
- The V4 regime structure appeared more suitable for ML-based trade selection. 

════════════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False

DATA_DIR = "Your_Local_Data_Path" # ml 이후 새로 뽑은 데이터 수집 파일.
FILES = {
    "ES": "ES_4h_continuous.csv",
    "NQ": "NQ_4h_continuous.csv",
    "YM": "YM_4h_continuous.csv",
    "RTY": "RTY_4h_continuous.csv",
}
PRODUCTS = list(FILES.keys()) # MyKey

# Data files are not included in this repository.
# Users must provide their own continuous futures datasets.

# 백테스트 파라미터 (동일 파라미터)
WINDOW = 100        # Regime 안정성 확보 - short-term noise 제거 목적.
COST = 0.0004       # 지수 선물 수수료 (왕복값)   
ATR_LEN = 14        # ATR 길이
BB_LENGTH = 20      # 볼린저밴드 길이 
STOP_ATR = 4.0      # TAIL-RISK 최소화
MOM_LOOKBACK = 10   # 모멘텀 전략 기준
V1_VR_LOWER = 0.95  # MEAN-REVERSION / TREND 구간 분리 기준1
V1_VR_UPPER = 1.05  # MEAN-REVERSION / TREND 구간 분리 기준2
V1_TRAIL_ATR = 3.0  
V4_TRAIL_MOM = 3.0

SHORT_Q = [2, 3, 4, 6, 8]      # q=1 : 4시간봉, q=2: 8시간봉 (단기봉 기준)  
LONG_Q = [10, 16, 21, 25, 30]  # 장기봉 기준
Q_LIST = SHORT_Q + LONG_Q      # 샘플 추가 목적.
ROLL_Q = 800                   """ regime 시장 상태 설명 위해 긴 기간 설정. + 작은 윈도우에서 노이즈 거르기 위함.
                               This is not a tuned hyperparameter but a structural sample size chosen to ensure that stable distributional estimation of VR across multi-q-values, and that reduced sensibility to short-term microstructure noise,
                               and that regime classification robustness under the non-stationary conditions. """

# BBW Regime Filter Layer (for Momentum Strategy) 
BBW_PERCENTILE_WINDOW = 100    # Percentile 안정 구간 - REGIME SWITCHING 기준 단기/중기 변동성 분포가 균형된 구간. (노이즈 X, LAG 적음)
EXPANSION_Q = 0.80             # BBW 기준 상위 20% 구간 for Regime Detection and signal frequency.
BAND_WALK_BARS = 5             # 더 긴 봉 구간 늘리기 가능. (5개 기준 - directional pressure 확인 가능하다 생각함.)
BAND_WALK_THRESHOLD = 3        # score threshold임. 3점 정도면 persistence okay!
BAND_WALK_SIGMA = 1.5          # Midpoint for Volatility-Normalized Activation Threshold. 

# Forward 경계
TRAIN_END = pd.Timestamp("2025-04-03 23:59:59")    # 학습 기간 (2018년 to 2025년)
TEST_START = pd.Timestamp("2025-04-04")            # 테스트 시작 기간.

# ML
THRESHOLD = 0.60      # edge가 보통 0.5 이상에서 나옴. 하지만, 0.55 to 0.6은 weakness함. 또한, false positive entry를 줄이기 위함임. 
SEEDS = [10, 100, 1000, 10000, 100000]   # For Seed-Robustness Validation. (Multi-Seeds) 


def variance_ratio(prices, q):
    log_p = np.log(prices); rets = np.diff(log_p); n = len(rets)
    if n < q + 1: return np.nan
    mu = np.mean(rets); var_1 = np.sum((rets - mu) ** 2) / n
    if var_1 == 0: return np.nan
    q_rets = log_p[q:] - log_p[:-q] # variance reduction + smoothing, 
    return np.sum((q_rets - q * mu) ** 2) / (n * q) / var_1 # normal 구조. 

    # It's for detect momentum and mean reversion regimes. 
    """ VR 에서 1이 의미하는 것:
    VR = 1 : 랜덤 워크, NO Auto-correlated
    VR > 1 : Momentum / Trend Persistence
    VR < 1 : Mean Reversion 
    """


# Multi-layer market state encoder.

""" 4차원 레짐 분해 
Layer 1 : 변동성 레짐 - bbw + expansion
Layer 2 : 추세 지속 - band walk
Layer 3 : 유효성 구조 - VR (multi-q-values)
Layer 4 : 레짐 정규화 - rolling quantiles of vr_score. 
"""
def prep(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["ma"] = df["close"].rolling(BB_LENGTH).mean()
    df["std"] = df["close"].rolling(BB_LENGTH).std()
    df["upper_2"] = df["ma"] + 2.0 * df["std"]
    df["lower_2"] = df["ma"] - 2.0 * df["std"]
    df["upper_walk"] = df["ma"] + BAND_WALK_SIGMA * df["std"]
    df["lower_walk"] = df["ma"] - BAND_WALK_SIGMA * df["std"]
    df["bbw"] = (df["upper_2"] - df["lower_2"]) / df["ma"]
    df["bbw_high_th"] = df["bbw"].rolling(BBW_PERCENTILE_WINDOW).quantile(EXPANSION_Q)
    df["is_expansion"] = (df["bbw"] > df["bbw_high_th"]).astype(int)
    df["bbw_percentile"] = df["bbw"].rolling(BBW_PERCENTILE_WINDOW).rank(pct=True)
    
    above = (df["close"] > df["upper_walk"]).astype(int)
    below = (df["close"] < df["lower_walk"]).astype(int)
    df["walk_up"] = (above.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD).astype(int)
    df["walk_down"] = (below.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD).astype(int)
    
    closes = df["close"].values; n = len(closes)
    for q in Q_LIST:
        arr = np.full(n, np.nan)
        for i in range(WINDOW, n):
            arr[i] = variance_ratio(closes[i - WINDOW:i], q)
        df[f"vr{q}"] = arr
    
    short_cols = [f"vr{q}" for q in SHORT_Q]
    long_cols = [f"vr{q}" for q in LONG_Q]
    df["short_vr"] = df[short_cols].mean(axis=1)
    df["long_vr"] = df[long_cols].mean(axis=1)
    df["vr_score"] = 0.8 * (1 - df["long_vr"]) + 0.2 * (1 - df["short_vr"])
    df["q20"] = df["vr_score"].rolling(ROLL_Q).quantile(0.20)
    df["q40"] = df["vr_score"].rolling(ROLL_Q).quantile(0.40)
    df["q60"] = df["vr_score"].rolling(ROLL_Q).quantile(0.60)
    df["q80"] = df["vr_score"].rolling(ROLL_Q).quantile(0.80)


  # 앞 내용 바탕으로 실전 트레이딩 입력 코드. 
  # vr_score to 5개 구간: strong_mom, mom, neutral, rev, and strong_rev. 
  # 한마디로, 연속적 regime 을 이산적인 시장 상태로 변환 후 다중 시점 모멘텀, 평균 회귀 거리, 시계열 구조 결합 => ML 활용.
  # 변동성 정규화 특성 공간 구축

    def reg5(row):
        if pd.isna(row["q20"]): return "neutral"
        s = row["vr_score"]
        if s <= row["q20"]: return "strong_mom"
        elif s <= row["q40"]: return "mom"
        elif s <= row["q60"]: return "neutral"
        elif s <= row["q80"]: return "rev"
        else: return "strong_rev"
    df["regime5"] = df.apply(reg5, axis=1)
    
    df["mom_val"] = df["close"] - df["close"].shift(MOM_LOOKBACK)
    df["mom_5"] = df["close"] - df["close"].shift(5)
    df["mom_20"] = df["close"] - df["close"].shift(20)
    
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    df["atr_norm"] = df["atr"] / df["close"]
    df["mom_5_norm"] = df["mom_5"] / df["atr"]
    df["mom_10_norm"] = df["mom_val"] / df["atr"]
    df["mom_20_norm"] = df["mom_20"] / df["atr"]
    df["bb_position"] = (df["close"] - df["lower_2"]) / (df["upper_2"] - df["lower_2"])
    df["dist_ma_norm"] = (df["close"] - df["ma"]) / df["atr"]
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_std_20"] = df["log_ret"].rolling(20).std()
    df["rolling_std_100"] = df["log_ret"].rolling(100).std()
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    
    critical = ["ma", "upper_2", "lower_2", "atr", "vr_score", "q20",
                "q40", "q60", "q80", "regime5", "mom_val", "vr16", "vr30",
                "rolling_std_100", "bbw_percentile"]
    return df.dropna(subset=critical).reset_index(drop=True)


def extract_features(row, product, source_model, direction, entry_type):
    return {
        "product": product, "source_model": source_model,
        "entry_dt": row["datetime"], "direction": direction,
        "entry_type": entry_type, "entry_price": row["close"], "entry_atr": row["atr"],
        "vr_16": row["vr16"], "vr_30": row["vr30"],
        "short_vr": row["short_vr"], "long_vr": row["long_vr"],
        "vr_score": row["vr_score"],
        "regime5_strong_mom": 1 if row["regime5"] == "strong_mom" else 0,
        "regime5_mom": 1 if row["regime5"] == "mom" else 0,
        "regime5_neutral": 1 if row["regime5"] == "neutral" else 0,
        "regime5_rev": 1 if row["regime5"] == "rev" else 0,
        "regime5_strong_rev": 1 if row["regime5"] == "strong_rev" else 0,
        "atr": row["atr"], "atr_norm": row["atr_norm"],
        "bbw": row["bbw"], "bbw_percentile": row["bbw_percentile"],
        "is_expansion": row["is_expansion"],
        "walk_up": row["walk_up"], "walk_down": row["walk_down"],
        "rolling_std_20": row["rolling_std_20"], "rolling_std_100": row["rolling_std_100"],
        "bb_position": row["bb_position"], "dist_ma_norm": row["dist_ma_norm"],
        "mom_5_norm": row["mom_5_norm"], "mom_10_norm": row["mom_10_norm"],
        "mom_20_norm": row["mom_20_norm"],
        "hour_of_day": row["hour_of_day"], "day_of_week": row["day_of_week"],
    }

"""모델 1 (V1)"""
def model_v1(df, product):
    pos, ep, sp, trail, m_pos = 0, 0.0, None, None, None
    entry_feat = None; trades = []
    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper_2"]; l = row["lower_2"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        vr_rev = row["vr16"] < V1_VR_LOWER and row["vr30"] < V1_VR_LOWER
        vr_tr = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
        bbw_long = row["is_expansion"] and row["walk_up"]
        bbw_short = row["is_expansion"] and row["walk_down"]
        if pos == 0:
            direction = 0; et = None
            if vr_rev:
                if c < l:
                    pos, ep, sp, m_pos = 1, c, c - STOP_ATR * atr, "rev"
                    direction, et = 1, "vr_rev"
                elif c > u:
                    pos, ep, sp, m_pos = -1, c, c + STOP_ATR * atr, "rev"
                    direction, et = -1, "vr_rev"
            elif vr_tr and mom != 0:
                d = 1 if mom > 0 else -1
                pos, ep, m_pos, trail = d, c, "trend", c
                sp = c - d * STOP_ATR * atr
                direction, et = d, "vr_trend"
            elif bbw_long:
                pos, ep, m_pos, trail = 1, c, "trend", c
                sp = c - STOP_ATR * atr; direction, et = 1, "bbw"
            elif bbw_short:
                pos, ep, m_pos, trail = -1, c, "trend", c
                sp = c + STOP_ATR * atr; direction, et = -1, "bbw"
            if pos != 0:
                entry_feat = extract_features(row, product, "v1", direction, et)
            continue
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                vr_tr_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_tr_now or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V1_TRAIL_ATR * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                vr_tr_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_tr_now or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V1_TRAIL_ATR * atr or not still: raw = (ep - c) / ep
        if raw is not None:
            entry_feat["exit_dt"] = dt; entry_feat["exit_price"] = c
            entry_feat["trade_return"] = raw - COST
            entry_feat["y"] = 1 if (raw - COST) > 0 else 0
            trades.append(entry_feat); pos = 0; entry_feat = None
    return trades

"""모델 4 (V4)"""
def model_v4(df, product):
    pos, ep, sp, trail, m_pos, entry_reg = 0, 0.0, None, None, None, None
    entry_feat = None; trades = []
    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper_2"]; l = row["lower_2"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        reg = row["regime5"]
        if pos == 0:
            entered = False; direction = 0; et = None
            if reg in ("strong_rev", "rev"):
                if c < l:
                    pos, ep, sp, m_pos, entry_reg = 1, c, c - STOP_ATR * atr, "rev", reg
                    direction, et, entered = 1, f"regime_{reg}", True
                elif c > u:
                    pos, ep, sp, m_pos, entry_reg = -1, c, c + STOP_ATR * atr, "rev", reg
                    direction, et, entered = -1, f"regime_{reg}", True
            elif reg == "strong_mom" and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg = 1, c, "trend", c, reg
                    sp = c - STOP_ATR * atr
                    direction, et, entered = 1, "strong_mom_bbw", True
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg = -1, c, "trend", c, reg
                    sp = c + STOP_ATR * atr
                    direction, et, entered = -1, "strong_mom_bbw", True
            if not entered and reg not in ("strong_rev", "rev", "strong_mom") and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg = 1, c, "trend", c, reg
                    sp = c - STOP_ATR * atr; direction, et = 1, "bbw_other"
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg = -1, c, "trend", c, reg
                    sp = c + STOP_ATR * atr; direction, et = -1, "bbw_other"
            if pos != 0:
                entry_feat = extract_features(row, product, "v4", direction, et)
            continue
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V4_TRAIL_MOM * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V4_TRAIL_MOM * atr or not still: raw = (ep - c) / ep
        if raw is not None:
            entry_feat["exit_dt"] = dt; entry_feat["exit_price"] = c
            entry_feat["trade_return"] = raw - COST
            entry_feat["y"] = 1 if (raw - COST) > 0 else 0
            trades.append(entry_feat); pos = 0; entry_feat = None; entry_reg = None
    return trades

# XGBOOST 모델
def get_xgb(seed):
    if USE_XGB:
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            use_label_encoder=False, eval_metric="logloss", verbosity=0)
    return GradientBoostingClassifier(n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=seed)


def stats(df_trades):
    if len(df_trades) == 0: return dict(n=0, sharpe=0, mdd=0, total=0, win_rate=0)
    rets = df_trades["trade_return"].values
    n = len(rets)
    years = (df_trades["exit_dt"].max() - df_trades["exit_dt"].min()).days / 365.25
    years = max(years, 0.01)
    sharpe = (rets.mean() / rets.std() * np.sqrt(n / years)) if rets.std() > 0 else 0
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    total = (np.prod(1 + rets) - 1) * 100
    return dict(n=n, sharpe=sharpe, mdd=mdd, total=total,
                win_rate=(rets > 0).mean() * 100)

# 메인 코드
def main():
    print("=" * 90)
    print(" Forward Validation — 2018-04-04 ~ 2025-04-03")
    print("=" * 90)
    print(f"\n  학습 기간: ~2026-04-03")
    print(f"  Test 기간: 2025-04-04 ~ 2026-06-10 (약 14개월)")
    
    # 1. 데이터 로드 + prep
    print("\n[1] 데이터 로드 + prep")
    all_trades_v1 = []
    all_trades_v4 = []
    
    for product, fn in FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print(f"  {path} 없음, 건너뜀"); continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        data = prep(df)
        print(f"  {product}: {len(data):,}봉, "
              f"{data['datetime'].iloc[0].strftime('%Y-%m-%d')} ~ "
              f"{data['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
        
        all_trades_v1.extend(model_v1(data, product))
        all_trades_v4.extend(model_v4(data, product))
    
    df_v1 = pd.DataFrame(all_trades_v1).sort_values("entry_dt").reset_index(drop=True)
    df_v4 = pd.DataFrame(all_trades_v4).sort_values("entry_dt").reset_index(drop=True)
    
    # 2. Train/Test 분할
    v1_train = df_v1[df_v1["entry_dt"] <= TRAIN_END].copy()
    v1_test = df_v1[df_v1["entry_dt"] >= TEST_START].copy()
    v4_train = df_v4[df_v4["entry_dt"] <= TRAIN_END].copy()
    v4_test = df_v4[df_v4["entry_dt"] >= TEST_START].copy()
    
    print(f"\n[2] 거래 수")
    print(f"  v1: Train {len(v1_train)}, Test {len(v1_test)}")
    print(f"  v4: Train {len(v4_train)}, Test {len(v4_test)}")
    
    if len(v1_test) == 0 and len(v4_test) == 0:
        print("\n  Test 거래 없음. 데이터 확인 필요")
        return
    
    # 3. Feature 준비
    CATEGORICAL = ["product", "entry_type"]
    ID_COLS = ["entry_dt", "exit_dt", "entry_price", "exit_price",
               "trade_return", "y", "source_model"]
    
    def prepare_xy(df_train, df_test):
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_enc = pd.get_dummies(df_all, columns=CATEGORICAL, drop_first=False)
        feature_cols = [c for c in df_enc.columns if c not in ID_COLS and c not in CATEGORICAL]
        
        n_train = len(df_train)
        X_train = df_enc[feature_cols].iloc[:n_train].astype(float).values
        y_train = df_enc["y"].iloc[:n_train].astype(int).values
        X_test = df_enc[feature_cols].iloc[n_train:].astype(float).values
        
        return X_train, y_train, X_test, feature_cols
    
    # 4. v1 ML 학습 + Forward
    print("\n" + "=" * 90)
    print("[3] v1 + XGBoost Forward Validation")
    print("=" * 90)
    
    v1_base_stats = stats(v1_test)
    print(f"\n  v1 베이스라인 (필터 X):")
    print(f"    거래 {v1_base_stats['n']}, 성공률 {v1_base_stats['win_rate']:.1f}%, "
          f"Sharpe {v1_base_stats['sharpe']:+.3f}, 수익 {v1_base_stats['total']:+.2f}%, "
          f"MDD {v1_base_stats['mdd']:+.1f}%")
    
    print(f"\n  v1 ML 필터 (5 seed):")
    print(f"  {'Seed':>8} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'수익':>9} {'MDD':>8}")
    print("  " + "─" * 55)
    
    v1_ml_results = []
    if len(v1_train) > 50 and len(v1_test) > 0:
        X_train, y_train, X_test, _ = prepare_xy(v1_train, v1_test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        for seed in SEEDS:
            model = get_xgb(seed)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            v1_test_copy = v1_test.copy()
            v1_test_copy["proba"] = proba
            filtered = v1_test_copy[v1_test_copy["proba"] >= THRESHOLD]
            s = stats(filtered)
            v1_ml_results.append(s)
            print(f"  {seed:>8} {s['n']:>5} {s['win_rate']:>6.1f}% {s['sharpe']:>+7.3f} "
                  f"{s['total']:>+8.2f}% {s['mdd']:>+7.1f}%")
    
    if v1_ml_results:
        v1_ml_avg = {
            "n": np.mean([r["n"] for r in v1_ml_results]),
            "sharpe": np.mean([r["sharpe"] for r in v1_ml_results]),
            "mdd": np.mean([r["mdd"] for r in v1_ml_results]),
            "total": np.mean([r["total"] for r in v1_ml_results]),
            "win_rate": np.mean([r["win_rate"] for r in v1_ml_results]),
        }
        print(f"\n  ★ v1 ML 평균: 거래 {v1_ml_avg['n']:.0f}, 성공률 {v1_ml_avg['win_rate']:.1f}%, "
              f"Sharpe {v1_ml_avg['sharpe']:+.3f}, 수익 {v1_ml_avg['total']:+.2f}%, "
              f"MDD {v1_ml_avg['mdd']:+.1f}%")
    
    # 5. v4 ML 학습 + Forward
    print("\n" + "=" * 90)
    print("[4] v4 + XGBoost Forward Validation")
    print("=" * 90)
    
    v4_base_stats = stats(v4_test)
    print(f"\n  v4 베이스라인 (필터 X):")
    print(f"    거래 {v4_base_stats['n']}, 성공률 {v4_base_stats['win_rate']:.1f}%, "
          f"Sharpe {v4_base_stats['sharpe']:+.3f}, 수익 {v4_base_stats['total']:+.2f}%, "
          f"MDD {v4_base_stats['mdd']:+.1f}%")
    
    print(f"\n  v4 ML 필터 (5 seed):")
    print(f"  {'Seed':>8} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'수익':>9} {'MDD':>8}")
    print("  " + "─" * 55)
    
    v4_ml_results = []
    if len(v4_train) > 50 and len(v4_test) > 0:
        X_train, y_train, X_test, _ = prepare_xy(v4_train, v4_test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        for seed in SEEDS:
            model = get_xgb(seed)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            v4_test_copy = v4_test.copy()
            v4_test_copy["proba"] = proba
            filtered = v4_test_copy[v4_test_copy["proba"] >= THRESHOLD]
            s = stats(filtered)
            v4_ml_results.append(s)
            print(f"  {seed:>8} {s['n']:>5} {s['win_rate']:>6.1f}% {s['sharpe']:>+7.3f} "
                  f"{s['total']:>+8.2f}% {s['mdd']:>+7.1f}%")
    
    if v4_ml_results:
        v4_ml_avg = {
            "n": np.mean([r["n"] for r in v4_ml_results]),
            "sharpe": np.mean([r["sharpe"] for r in v4_ml_results]),
            "mdd": np.mean([r["mdd"] for r in v4_ml_results]),
            "total": np.mean([r["total"] for r in v4_ml_results]),
            "win_rate": np.mean([r["win_rate"] for r in v4_ml_results]),
        }
        print(f"\n  ★ v4 ML 평균: 거래 {v4_ml_avg['n']:.0f}, 성공률 {v4_ml_avg['win_rate']:.1f}%, "
              f"Sharpe {v4_ml_avg['sharpe']:+.3f}, 수익 {v4_ml_avg['total']:+.2f}%, "
              f"MDD {v4_ml_avg['mdd']:+.1f}%")
    
    # 6. 최종 비교
    print("\n" + "=" * 90)
    print("[5] 최종 — Forward Validation 결과")
    print("=" * 90)
    print(f"\n  {'구성':<25} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'14개월 수익':>10} {'MDD':>8}")
    print("  " + "─" * 70)
    print(f"  {'v1 베이스라인':<25} {v1_base_stats['n']:>5} {v1_base_stats['win_rate']:>6.1f}% "
          f"{v1_base_stats['sharpe']:>+7.3f} {v1_base_stats['total']:>+9.2f}% {v1_base_stats['mdd']:>+7.1f}%")
    if v1_ml_results:
        print(f"  {'v1 + XGBoost (평균)':<25} {v1_ml_avg['n']:>5.0f} {v1_ml_avg['win_rate']:>6.1f}% "
              f"{v1_ml_avg['sharpe']:>+7.3f} {v1_ml_avg['total']:>+9.2f}% {v1_ml_avg['mdd']:>+7.1f}%")
    print(f"  {'v4 베이스라인':<25} {v4_base_stats['n']:>5} {v4_base_stats['win_rate']:>6.1f}% "
          f"{v4_base_stats['sharpe']:>+7.3f} {v4_base_stats['total']:>+9.2f}% {v4_base_stats['mdd']:>+7.1f}%")
    if v4_ml_results:
        print(f"  {'v4 + XGBoost (평균)':<25} {v4_ml_avg['n']:>5.0f} {v4_ml_avg['win_rate']:>6.1f}% "
              f"{v4_ml_avg['sharpe']:>+7.3f} {v4_ml_avg['total']:>+9.2f}% {v4_ml_avg['mdd']:>+7.1f}%")


if __name__ == "__main__":
    main()
