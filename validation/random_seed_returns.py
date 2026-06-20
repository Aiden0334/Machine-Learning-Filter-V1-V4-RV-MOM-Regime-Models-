"""
═══════════════════════════════════════════════════════════════════
  분리 모델 정확한 수익률 측정
═══════════════════════════════════════════════════════════════════
[목적]
  v1 단독, v4 단독 각각의 정확한 4년 누적 수익률 측정
  5 seed 평균 (정직한 평가)
═══════════════════════════════════════════════════════════════════
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

INPUT_FILE = "./ml/features.csv"
ID_COLS = ["entry_dt", "exit_dt", "entry_price", "exit_price",
           "trade_return", "y"]
CATEGORICAL_COLS = ["product", "source_model", "entry_type"]
DATA_END = pd.Timestamp("2025-12-31 23:59:59")
THRESHOLD = 0.60
SEEDS = [10, 100, 1000, 10000, 100000]

FOLDS = [
    {"train_end": "2021-12-31", "test_year": 2022},
    {"train_end": "2022-12-31", "test_year": 2023},
    {"train_end": "2023-12-31", "test_year": 2024},
    {"train_end": "2024-12-31", "test_year": 2025},
]


def get_xgb(seed):
    if USE_XGB:
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            use_label_encoder=False, eval_metric="logloss", verbosity=0)
    return GradientBoostingClassifier(n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=seed)


def stats(df_trades):
    if len(df_trades) == 0: return dict(n=0, sharpe=0, mdd=0, total=0, cagr=0)
    rets = df_trades["trade_return"].values
    n = len(rets)
    years = (df_trades["exit_dt"].max() - df_trades["exit_dt"].min()).days / 365.25
    years = max(years, 0.01)
    sharpe = (rets.mean() / rets.std() * np.sqrt(n / years)) if rets.std() > 0 else 0
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    total = (np.prod(1 + rets) - 1) * 100
    cagr = ((1 + total/100) ** (1/years) - 1) * 100 if total > -100 else -100
    return dict(n=n, sharpe=sharpe, mdd=mdd, total=total, cagr=cagr, years=years)


def run_wf(df_data, feature_cols, seed):
    """4폴드 WF, XGBoost 필터 결과"""
    all_base = []
    all_filt = []
    
    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"] + " 23:59:59")
        test_end = pd.Timestamp(f"{fold['test_year']}-12-31 23:59:59")
        df_train = df_data[df_data["entry_dt"] <= train_end].copy()
        df_test = df_data[(df_data["entry_dt"] > train_end) & 
                          (df_data["entry_dt"] <= test_end)].copy()
        if len(df_train) < 100 or len(df_test) < 15: continue
        
        X_train = df_train[feature_cols].astype(float).values
        y_train = df_train["y"].astype(int).values
        X_test = df_test[feature_cols].astype(float).values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = get_xgb(seed)
        model.fit(X_train_s, y_train)
        
        df_test["proba"] = model.predict_proba(X_test_s)[:, 1]
        all_base.extend(df_test.to_dict("records"))
        all_filt.extend(df_test[df_test["proba"] >= THRESHOLD].to_dict("records"))
    
    return pd.DataFrame(all_base), pd.DataFrame(all_filt)


def main():
    print("=" * 90)
    print(" 분리 모델 정확한 수익률 측정 (5 seed 평균)")
    print("=" * 90)
    
    df = pd.read_csv(INPUT_FILE, parse_dates=["entry_dt", "exit_dt"])
    df = df[df["entry_dt"] <= DATA_END].copy().reset_index(drop=True)
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    feature_cols = [c for c in df_enc.columns 
                    if c not in ID_COLS and c not in CATEGORICAL_COLS
                    and not c.startswith("source_model_")]
    df_enc["product"] = df["product"]
    df_enc["source_model"] = df["source_model"]
    
    df_v1 = df_enc[df_enc["source_model"] == "v1"].copy().reset_index(drop=True)
    df_v4 = df_enc[df_enc["source_model"] == "v4"].copy().reset_index(drop=True)
    
    # 베이스라인 (각각)
    v1_base, _ = run_wf(df_v1, feature_cols, 42)  # 베이스는 seed 무관
    v4_base, _ = run_wf(df_v4, feature_cols, 42)
    s_v1_base = stats(v1_base)
    s_v4_base = stats(v4_base)
    
    # 각 seed별 측정
    print(f"\n[v1 단독 운용 — XGBoost 필터]")
    print(f"  베이스라인: 거래 {s_v1_base['n']}, Sharpe {s_v1_base['sharpe']:+.3f}, "
          f"MDD {s_v1_base['mdd']:+.1f}%, 수익 {s_v1_base['total']:+.2f}%, "
          f"CAGR {s_v1_base['cagr']:+.2f}%")
    print(f"\n  {'Seed':>8} {'거래':>5} {'Sharpe':>8} {'MDD':>8} {'4년수익':>10} {'CAGR':>8}")
    print("  " + "─" * 60)
    
    v1_results = []
    for seed in SEEDS:
        _, filt = run_wf(df_v1, feature_cols, seed)
        s = stats(filt)
        v1_results.append(s)
        print(f"  {seed:>8} {s['n']:>5} {s['sharpe']:>+7.3f} {s['mdd']:>+7.1f}% "
              f"{s['total']:>+9.2f}% {s['cagr']:>+7.2f}%")
    
    v1_avg = {
        "n": np.mean([r["n"] for r in v1_results]),
        "sharpe": np.mean([r["sharpe"] for r in v1_results]),
        "mdd": np.mean([r["mdd"] for r in v1_results]),
        "total": np.mean([r["total"] for r in v1_results]),
        "cagr": np.mean([r["cagr"] for r in v1_results]),
    }
    print(f"\n  ★ 5 seed 평균:")
    print(f"     거래 {v1_avg['n']:.0f}, Sharpe {v1_avg['sharpe']:+.3f}, "
          f"MDD {v1_avg['mdd']:+.1f}%, 4년 수익 {v1_avg['total']:+.2f}%, "
          f"CAGR {v1_avg['cagr']:+.2f}%")
    
    print(f"\n[v4 단독 운용 — XGBoost 필터]")
    print(f"  베이스라인: 거래 {s_v4_base['n']}, Sharpe {s_v4_base['sharpe']:+.3f}, "
          f"MDD {s_v4_base['mdd']:+.1f}%, 수익 {s_v4_base['total']:+.2f}%, "
          f"CAGR {s_v4_base['cagr']:+.2f}%")
    print(f"\n  {'Seed':>8} {'거래':>5} {'Sharpe':>8} {'MDD':>8} {'4년수익':>10} {'CAGR':>8}")
    print("  " + "─" * 60)
    
    v4_results = []
    for seed in SEEDS:
        _, filt = run_wf(df_v4, feature_cols, seed)
        s = stats(filt)
        v4_results.append(s)
        print(f"  {seed:>8} {s['n']:>5} {s['sharpe']:>+7.3f} {s['mdd']:>+7.1f}% "
              f"{s['total']:>+9.2f}% {s['cagr']:>+7.2f}%")
    
    v4_avg = {
        "n": np.mean([r["n"] for r in v4_results]),
        "sharpe": np.mean([r["sharpe"] for r in v4_results]),
        "mdd": np.mean([r["mdd"] for r in v4_results]),
        "total": np.mean([r["total"] for r in v4_results]),
        "cagr": np.mean([r["cagr"] for r in v4_results]),
    }
    print(f"\n  ★ 5 seed 평균:")
    print(f"     거래 {v4_avg['n']:.0f}, Sharpe {v4_avg['sharpe']:+.3f}, "
          f"MDD {v4_avg['mdd']:+.1f}%, 4년 수익 {v4_avg['total']:+.2f}%, "
          f"CAGR {v4_avg['cagr']:+.2f}%")
    
    # ═══════════════════════════════════════
    # 최종 정리
    # ═══════════════════════════════════════
    print("\n" + "=" * 90)
    print("[최종 — 정직한 평가 (5 seed 평균)]")
    print("=" * 90)
    
    print(f"\n  {'구성':<25} {'거래':>5} {'Sharpe':>8} {'MDD':>8} {'4년수익':>10} {'CAGR':>8}")
    print("  " + "─" * 75)
    print(f"  {'v1 베이스라인':<25} {s_v1_base['n']:>5} {s_v1_base['sharpe']:>+7.3f} "
          f"{s_v1_base['mdd']:>+7.1f}% {s_v1_base['total']:>+9.2f}% {s_v1_base['cagr']:>+7.2f}%")
    print(f"  {'v1 ML 필터 (XGBoost)':<25} {v1_avg['n']:>5.0f} {v1_avg['sharpe']:>+7.3f} "
          f"{v1_avg['mdd']:>+7.1f}% {v1_avg['total']:>+9.2f}% {v1_avg['cagr']:>+7.2f}%")
    print(f"  {'v4 베이스라인':<25} {s_v4_base['n']:>5} {s_v4_base['sharpe']:>+7.3f} "
          f"{s_v4_base['mdd']:>+7.1f}% {s_v4_base['total']:>+9.2f}% {s_v4_base['cagr']:>+7.2f}%")
    print(f"  {'v4 ML 필터 (XGBoost)':<25} {v4_avg['n']:>5.0f} {v4_avg['sharpe']:>+7.3f} "
          f"{v4_avg['mdd']:>+7.1f}% {v4_avg['total']:>+9.2f}% {v4_avg['cagr']:>+7.2f}%")


if __name__ == "__main__":
    main()
