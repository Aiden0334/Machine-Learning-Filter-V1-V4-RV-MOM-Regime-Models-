"""
═══════════════════════════════════════════════════════════════════
  ML Step 2 — XGBoost Walk-Forward
═══════════════════════════════════════════════════════════════════
[Step 1 결과 (Logistic)]
  4폴드 합산 Sharpe: +0.562 개선 (베이스 1.021 → 1.583)
  3/4 폴드 개선
  MDD -55.5% → -34.9%

[Step 2 목표]
  XGBoost로 비선형 + 상호작용 효과 측정
  Logistic 대비 추가 개선 측정
  Feature importance로 핵심 변수 파악

[설정]
  Walk-Forward 4폴드 (2026 제외)
  고정 임계 0.60 (Step 1과 동일, 비교 가능성)
  보수적 hyperparameter (단순함의 가치)
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# XGBoost 시도, 없으면 sklearn GradientBoosting
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("⚠ xgboost 없음, sklearn GradientBoosting 사용")

INPUT_FILE = "./ml/features.csv"

ID_COLS = ["entry_dt", "exit_dt", "entry_price", "exit_price",
           "trade_return", "y"]
CATEGORICAL_COLS = ["product", "source_model", "entry_type"]

DATA_END = pd.Timestamp("2025-12-31 23:59:59")

FOLDS = [
    {"name": "Fold 1", "train_end": "2021-12-31", "test_year": 2022},
    {"name": "Fold 2", "train_end": "2022-12-31", "test_year": 2023},
    {"name": "Fold 3", "train_end": "2023-12-31", "test_year": 2024},
    {"name": "Fold 4", "train_end": "2024-12-31", "test_year": 2025},
]

FIXED_THRESHOLD = 0.60


def get_model():
    """XGBoost 또는 fallback"""
    if USE_XGB:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )


def compute_sharpe(returns, years):
    if len(returns) == 0 or years <= 0: return 0
    rets = np.array(returns)
    if rets.std() == 0: return 0
    return rets.mean() / rets.std() * np.sqrt(len(rets) / years)


def trading_stats(df_trades):
    if len(df_trades) == 0:
        return dict(n=0, win_rate=0, mean_ret=0, sharpe=0, mdd=0, total=0)
    rets = df_trades["trade_return"].values
    n = len(rets)
    years = (df_trades["exit_dt"].max() - df_trades["exit_dt"].min()).days / 365.25
    years = max(years, 0.01)
    win_rate = (rets > 0).mean() * 100
    mean_ret = rets.mean() * 100
    sharpe = compute_sharpe(rets, years)
    total = (np.prod(1 + rets) - 1) * 100
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    return dict(n=n, win_rate=win_rate, mean_ret=mean_ret,
                sharpe=sharpe, mdd=mdd, total=total)


def main():
    print("=" * 100)
    model_name = "XGBoost" if USE_XGB else "GradientBoosting"
    print(f" ML Step 2 — {model_name} Walk-Forward (4폴드, 2026 제외)")
    print("=" * 100)
    
    df = pd.read_csv(INPUT_FILE, parse_dates=["entry_dt", "exit_dt"])
    df = df[df["entry_dt"] <= DATA_END].copy().reset_index(drop=True)
    
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    feature_cols = [c for c in df_enc.columns 
                    if c not in ID_COLS and c not in CATEGORICAL_COLS]
    df_enc["product"] = df["product"]
    df_enc["source_model"] = df["source_model"]
    
    print(f"\n[데이터] {len(df_enc):,} 거래, {len(feature_cols)} feature")
    
    # ═══════════════════════════════════════
    # Walk-Forward (고정 임계 0.60)
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print(f"[A] Walk-Forward — 고정 임계 {FIXED_THRESHOLD}")
    print("=" * 100)
    print(f"  {'Fold':<8} {'Test':>6} {'AUC':>6} "
          f"{'베이스라인':>26} {'필터링후':>26} {'개선':>9}")
    print(f"  {'':<8} {'':>6} {'':>6} "
          f"{'n  성공률  Sharpe  MDD':<26} {'n  성공률  Sharpe  MDD':<26}")
    print("  " + "─" * 90)
    
    results = []
    all_baseline = []
    all_filtered = []
    feature_importances_list = []
    
    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"] + " 23:59:59")
        test_end = pd.Timestamp(f"{fold['test_year']}-12-31 23:59:59")
        
        df_train = df_enc[df_enc["entry_dt"] <= train_end].copy()
        df_test = df_enc[(df_enc["entry_dt"] > train_end) & 
                          (df_enc["entry_dt"] <= test_end)].copy()
        
        if len(df_train) < 200 or len(df_test) < 30:
            print(f"  {fold['name']:<8} 표본 부족"); continue
        
        X_train = df_train[feature_cols].astype(float).values
        y_train = df_train["y"].astype(int).values
        X_test = df_test[feature_cols].astype(float).values
        y_test = df_test["y"].astype(int).values
        
        # 정규화 (XGBoost는 사실 필요 없지만, 일관성 위해)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 학습
        model = get_model()
        model.fit(X_train_s, y_train)
        
        # 예측
        test_proba = model.predict_proba(X_test_s)[:, 1]
        df_test["proba"] = test_proba
        
        # AUC
        try:
            auc = roc_auc_score(y_test, test_proba)
        except:
            auc = 0
        
        # 평가
        baseline = trading_stats(df_test)
        filtered = trading_stats(df_test[df_test["proba"] >= FIXED_THRESHOLD])
        sharpe_delta = filtered["sharpe"] - baseline["sharpe"]
        
        print(f"  {fold['name']:<8} {fold['test_year']:>6} {auc:>6.3f} "
              f"{baseline['n']:>3} {baseline['win_rate']:>5.1f}% {baseline['sharpe']:>+5.2f} {baseline['mdd']:>+6.1f}%  "
              f"{filtered['n']:>3} {filtered['win_rate']:>5.1f}% {filtered['sharpe']:>+5.2f} {filtered['mdd']:>+6.1f}%  "
              f"{sharpe_delta:>+7.3f}")
        
        results.append({
            "fold": fold["name"], "test_year": fold["test_year"],
            "auc": auc, "delta": sharpe_delta,
            "baseline_sharpe": baseline["sharpe"],
            "filtered_sharpe": filtered["sharpe"]
        })
        
        all_baseline.extend(df_test.to_dict("records"))
        all_filtered.extend(df_test[df_test["proba"] >= FIXED_THRESHOLD].to_dict("records"))
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            feature_importances_list.append(model.feature_importances_)
    
    if results:
        deltas = np.array([r["delta"] for r in results])
        aucs = np.array([r["auc"] for r in results])
        print(f"\n  Sharpe 개선 분포:")
        print(f"    평균: {deltas.mean():+.3f}, std {deltas.std():.3f}, "
              f"범위 [{deltas.min():+.3f}, {deltas.max():+.3f}]")
        print(f"    개선 폴드: {(deltas > 0).sum()}/{len(deltas)}")
        print(f"    평균 AUC:  {aucs.mean():.3f}")
    
    # ═══════════════════════════════════════
    # [B] 4폴드 합산
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[B] 4폴드 합산 (Test 2022-2025)")
    print("=" * 100)
    
    df_all_base = pd.DataFrame(all_baseline)
    df_all_filt = pd.DataFrame(all_filtered)
    
    s_base = trading_stats(df_all_base)
    s_filt = trading_stats(df_all_filt)
    sharpe_delta = s_filt["sharpe"] - s_base["sharpe"]
    
    print(f"\n  {'구성':<25} {'거래':>5} {'성공률':>7} {'평균수익':>10} {'Sharpe':>8} {'MDD':>8} {'총수익':>10}")
    print("  " + "─" * 82)
    print(f"  {'베이스라인':<25} {s_base['n']:>5} {s_base['win_rate']:>6.1f}% "
          f"{s_base['mean_ret']:>+9.3f}% {s_base['sharpe']:>+7.3f} {s_base['mdd']:>+7.1f}% {s_base['total']:>+9.2f}%")
    print(f"  {model_name + ' 필터':<25} {s_filt['n']:>5} {s_filt['win_rate']:>6.1f}% "
          f"{s_filt['mean_ret']:>+9.3f}% {s_filt['sharpe']:>+7.3f} {s_filt['mdd']:>+7.1f}% {s_filt['total']:>+9.2f}%")
    
    print(f"\n  변화: Sharpe {sharpe_delta:+.3f}, "
          f"거래 거부 {s_base['n']-s_filt['n']} ({(s_base['n']-s_filt['n'])/s_base['n']*100:.1f}%)")
    
    # ═══════════════════════════════════════
    # [C] Logistic과 비교
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[C] Step 1 (Logistic) vs Step 2 (" + model_name + ") 비교")
    print("=" * 100)
    
    print(f"\n  {'구성':<30} {'Sharpe':>8} {'MDD':>8} {'총수익':>10}")
    print("  " + "─" * 60)
    print(f"  {'베이스라인 (필터 없음)':<30} {s_base['sharpe']:>+7.3f} "
          f"{s_base['mdd']:>+7.1f}% {s_base['total']:>+9.2f}%")
    print(f"  {'Step 1 Logistic (고정 0.60)':<30} {'+1.583':>8} {'-34.9%':>8} {'+352.41%':>10}")
    print(f"  {'Step 2 ' + model_name + ' (고정 0.60)':<30} {s_filt['sharpe']:>+7.3f} "
          f"{s_filt['mdd']:>+7.1f}% {s_filt['total']:>+9.2f}%")
    
    xgb_vs_logistic = s_filt["sharpe"] - 1.583
    print(f"\n  XGBoost vs Logistic: Sharpe {xgb_vs_logistic:+.3f}")
    
    # ═══════════════════════════════════════
    # [D] 종목/모델별
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print(f"[D] 종목/모델별 (4폴드 합산, 고정 {FIXED_THRESHOLD})")
    print("=" * 100)
    
    print(f"\n  종목별:")
    print(f"  {'종목':<5} {'베이스라인':>22} {'필터링후':>22} {'개선':>9}")
    print("  " + "─" * 60)
    for p in ["ES", "NQ", "YM", "RTY"]:
        sb = df_all_base[df_all_base["product"] == p]
        sf = df_all_filt[df_all_filt["product"] == p] if len(df_all_filt) > 0 else pd.DataFrame()
        s_b = trading_stats(sb)
        s_f = trading_stats(sf) if len(sf) > 0 else dict(n=0, win_rate=0, sharpe=0)
        delta = s_f["sharpe"] - s_b["sharpe"]
        print(f"  {p:<5} "
              f"{s_b['n']:>3} {s_b['win_rate']:>5.1f}% {s_b['sharpe']:>+6.3f}      "
              f"{s_f['n']:>3} {s_f['win_rate']:>5.1f}% {s_f['sharpe']:>+6.3f}      "
              f"{delta:>+6.3f}")
    
    print(f"\n  모델별:")
    print(f"  {'모델':<5} {'베이스라인':>22} {'필터링후':>22} {'개선':>9}")
    print("  " + "─" * 60)
    for m in ["v1", "v4"]:
        sb = df_all_base[df_all_base["source_model"] == m]
        sf = df_all_filt[df_all_filt["source_model"] == m] if len(df_all_filt) > 0 else pd.DataFrame()
        s_b = trading_stats(sb)
        s_f = trading_stats(sf) if len(sf) > 0 else dict(n=0, win_rate=0, sharpe=0)
        delta = s_f["sharpe"] - s_b["sharpe"]
        print(f"  {m:<5} "
              f"{s_b['n']:>3} {s_b['win_rate']:>5.1f}% {s_b['sharpe']:>+6.3f}      "
              f"{s_f['n']:>3} {s_f['win_rate']:>5.1f}% {s_f['sharpe']:>+6.3f}      "
              f"{delta:>+6.3f}")
    
    # ═══════════════════════════════════════
    # [E] Feature Importance
    # ═══════════════════════════════════════
    if feature_importances_list:
        print("\n" + "=" * 100)
        print(f"[E] Feature Importance (4폴드 평균 Top 15)")
        print("=" * 100)
        
        avg_importance = np.mean(feature_importances_list, axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": avg_importance
        }).sort_values("importance", ascending=False)
        
        for i, row in importance_df.head(15).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")
    
    # ═══════════════════════════════════════
    # 최종 판단
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[최종 판단]")
    print("=" * 100)
    
    if xgb_vs_logistic > 0.2:
        verdict = f"{model_name}이 Logistic보다 명확히 좋음 (+{xgb_vs_logistic:.3f}) → 비선형 정보 존재"
    elif xgb_vs_logistic > 0.05:
        verdict = f"{model_name}이 Logistic보다 살짝 좋음 (+{xgb_vs_logistic:.3f})"
    elif xgb_vs_logistic > -0.05:
        verdict = f"{model_name} ≈ Logistic ({xgb_vs_logistic:+.3f}) → 비선형 정보 부족"
    else:
        verdict = f"{model_name}이 Logistic보다 나쁨 ({xgb_vs_logistic:+.3f}) → 과적합 가능"
    
    print(f"\n  ▶ {verdict}")
    print(f"\n  다음 단계 결정 가이드:")
    print(f"    - XGBoost 명확히 더 좋음 → Step 3 (분리 모델) 또는 GRU 가치 있음")
    print(f"    - XGBoost ≈ Logistic → 천장 도달, 종료 권장")
    print(f"    - XGBoost 나쁨 → Logistic 채택, 종료")


if __name__ == "__main__":
    main()
