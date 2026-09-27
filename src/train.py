from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss
)

from .features import (
    load_data, clean_orders, make_train_features, make_train_val_features, add_base_features, add_fitted_history_rates,
    FEATURE_COLS, CATEGORICAL_COLS, HIST_ENTITIES, ROOT
)

MODEL_PATH = ROOT / "model.pkl"
PRED_PATH = ROOT / "predictions.csv"
REPORT_PATH = ROOT / "reports" / "evidence.md"

RETURN_COST = 1150.0
CALL_COST = 45.0
CALL_PREVENT_RATE = 0.35
HOLD_CANCEL_RATE = 0.12


def fit_model(X, y, cat_cols, iterations=150, eval_set=None):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y, cat_features=cat_idx, eval_set=eval_set, use_best_model=eval_set is not None,
              early_stopping_rounds=50 if eval_set is not None else None, verbose=False)
    return model


def prepare_xy(df):
    X = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].fillna("MISSING").astype(str)
    return X


def metrics_at_threshold(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "flags": int(pred.sum()), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        # Pilot-based savings proxy: only the stated 35% of true returns are preventable by a call.
        "call_net_value": float(tp * CALL_PREVENT_RATE * RETURN_COST - pred.sum() * CALL_COST),
    }


def main():
    print("Loading data...")
    train_raw, test_raw, customers, products = load_data()
    train_raw = clean_orders(train_raw, products, is_train=True)
    test_raw = clean_orders(test_raw, products, is_train=False)
    if test_raw.order_id.duplicated().any():
        raise ValueError("test_unlabelled.csv contains duplicate order_id values")

    train_month = train_raw["order_placed_at"].dt.to_period("M")
    months = sorted(train_month.unique())
    val_months = months[-5:]
    print(f"Unique train orders: {len(train_raw)}; returns: {train_raw.returned.sum()} ({train_raw.returned.mean():.2%})")
    print(f"Validation months: {[str(m) for m in val_months]}")

    oof_frames = []
    fold_rows = []
    best_iters = []
    for m in val_months:
        mask_train = train_raw["order_placed_at"].dt.to_period("M") < m
        mask_val = train_raw["order_placed_at"].dt.to_period("M") == m
        tr_part = train_raw.loc[mask_train].copy()
        va_part = train_raw.loc[mask_val].copy()
        tr_f, va_f = make_train_val_features(tr_part, va_part, customers, products)
        Xtr, Xv = prepare_xy(tr_f), prepare_xy(va_f)
        ytr, yv = tr_f["returned"].astype(int), va_f["returned"].astype(int)
        model = fit_model(Xtr, ytr, CATEGORICAL_COLS, iterations=500, eval_set=(Xv, yv))
        p = model.predict_proba(Xv)[:, 1]
        best_iters.append(max(50, model.get_best_iteration() + 1))
        fold_rows.append({
            "month": str(m), "n": len(yv), "positives": int(yv.sum()),
            "auc": roc_auc_score(yv, p), "ap": average_precision_score(yv, p),
            "best_iter": int(best_iters[-1]),
        })
        oof = va_f[["order_id", "returned"]].copy()
        oof["raw_score"] = p
        oof_frames.append(oof)
        print(f"{m}: AUC={fold_rows[-1]['auc']:.4f}, AP={fold_rows[-1]['ap']:.4f}, best_iter={best_iters[-1]}")

    oof = pd.concat(oof_frames, ignore_index=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["raw_score"], oof["returned"])
    oof["score"] = iso.transform(oof["raw_score"])

    auc_mean = float(oof.groupby("order_id").first().shape[0])  # rows count for report only
    cv_auc_mean = float(np.mean([x["auc"] for x in fold_rows]))
    cv_auc_std = float(np.std([x["auc"] for x in fold_rows]))
    cv_ap_mean = float(np.mean([x["ap"] for x in fold_rows]))
    cv_ap_std = float(np.std([x["ap"] for x in fold_rows]))

    thresholds = [0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    threshold_rows = [metrics_at_threshold(oof.returned.values, oof.score.values, t) for t in thresholds]
    best_call = max(threshold_rows, key=lambda x: x["call_net_value"])

    # For a classification-quality reference, find the highest OOF accuracy threshold.
    best_accuracy = max(threshold_rows, key=lambda x: x["accuracy"])

    # Final features: each train row uses only earlier rows for historical-rate features.
    final_train = make_train_features(train_raw, customers, products)
    final_test_base = add_base_features(test_raw, customers, products)
    final_test = add_fitted_history_rates(final_train, final_test_base, HIST_ENTITIES)

    Xfull, Xtest = prepare_xy(final_train), prepare_xy(final_test)
    yfull = final_train["returned"].astype(int)
    final_iterations = int(np.median(best_iters))
    final_model = fit_model(Xfull, yfull, CATEGORICAL_COLS, iterations=final_iterations)

    raw_test = final_model.predict_proba(Xtest)[:, 1]
    test_score = iso.transform(raw_test)
    submission = pd.DataFrame({"order_id": test_raw["order_id"].values, "score": test_score})
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or len(submission) != len(sample):
        raise ValueError("Prediction shape does not match sample_submission.csv")
    submission.to_csv(PRED_PATH, index=False)

    # Model-side feature importance; SHAP is performed at request time in the app for reasons.
    fi = final_model.get_feature_importance(Pool(Xfull, yfull, cat_features=[Xfull.columns.get_loc(c) for c in CATEGORICAL_COLS]),
                                            type="FeatureImportance")
    top_features = [name for name, imp in sorted(zip(FEATURE_COLS, fi), key=lambda x: -x[1])[:12]]

    model_artifact = {
        "model": final_model,
        "calibrator": iso,
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "call_threshold": float(best_call["threshold"]),
        "hold_threshold": 0.40,
        "thresholds_checked": thresholds,
        "cv_folds": fold_rows,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "cv_ap_mean": cv_ap_mean,
        "cv_ap_std": cv_ap_std,
        "oof_brier": float(brier_score_loss(oof["returned"], oof["score"])),
        "top_features": top_features,
        "final_iterations": final_iterations,
        "costs": {"return_cost": RETURN_COST, "call_cost": CALL_COST,
                  "call_prevent_rate": CALL_PREVENT_RATE, "hold_cancel_rate": HOLD_CANCEL_RATE},
    }
    joblib.dump(model_artifact, MODEL_PATH)

    # Evidence report.
    report_lines = [
        "# Evidence — Kestrel Returns Risk",
        "",
        "## Dataset and validation",
        f"- Training rows after partner re-import deduplication: {len(train_raw):,} unique orders.",
        f"- Historical return rate: {train_raw.returned.mean():.2%} ({int(train_raw.returned.sum()):,} returns).",
        f"- Temporal validation: five expanding-window monthly folds on {', '.join(str(x) for x in val_months)}.",
        "- Target-derived historical-rate features are recomputed inside each fold using earlier training data only.",
        "- `pickup_scheduled_at` and `last_service_event_type` are excluded because the operations policy says they are written during the return process.",
        "",
        "## Cross-validation",
        "| Validation month | Orders | Returns | ROC-AUC | Average Precision | Best iterations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in fold_rows:
        report_lines.append(f"| {r['month']} | {r['n']:,} | {r['positives']:,} | {r['auc']:.3f} | {r['ap']:.3f} | {r['best_iter']} |")
    report_lines += [
        f"| **Mean** | | | **{cv_auc_mean:.3f} ± {cv_auc_std:.3f}** | **{cv_ap_mean:.3f} ± {cv_ap_std:.3f}** | |",
        "",
        "### Pre-submission expectation",
        f"Expected held-out ranking performance: ROC-AUC ≈ **{cv_auc_mean:.3f}** and average precision ≈ **{cv_ap_mean:.3f}**, based only on the five temporal validation folds.",
        "",
        "## Threshold analysis (out-of-fold, calibrated scores)",
        "A confirmation call costs ₹45 and the pilot reported preventing about 35% of returns on called orders. The policy gives a ₹1,150 average return cost. The table therefore uses `TP × 35% × ₹1,150 − flags × ₹45` as an indicative call-pilot value. It does **not** invent a cancellation loss from order value.",
        "",
        "| Threshold | Accuracy | Precision | Recall | Flags | Indicative call-pilot value / validation sample |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in threshold_rows:
        report_lines.append(f"| {r['threshold']:.3f} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['flags']:,} | ₹{r['call_net_value']:,.0f} |")
    report_lines += [
        "",
        f"**Selected threshold:** {best_call['threshold']:.3f}, because it has the highest indicative call-pilot value on the pooled out-of-fold sample. This is an operating threshold for a call-first pilot, not a claim that every flagged order should automatically be held.",
        f"**Highest observed OOF accuracy among checked thresholds:** {best_accuracy['accuracy']:.3f} at threshold {best_accuracy['threshold']:.3f}. This is reported with recall/precision because 88.6% of historical orders were non-returns.",
        f"OOF Brier score after isotonic calibration: {model_artifact['oof_brier']:.4f}.",
        "",
        "## Data-quality decisions",
        "- 651 partner-feed re-imports were deduplicated by `order_id`, keeping the CRM copy.",
        "- October 2025 order values with an exact 100× merchandise-value ratio were divided by 100, matching the payment-gateway issue noted in the email thread.",
        "- `delivery_pincode = 000000` is retained as an explicit walk-in/no-address signal, per the data pack.",
        "",
        "## Failure modes / limitations",
        "- Returns are only ~11% of orders, so accuracy alone overstates usefulness.",
        "- The model cannot see every real-world reason for return; text is represented only through simple deterministic note signals.",
        "- The policy gives a 12% cancellation rate for holds over 24 hours but does not specify the financial loss per cancellation, so that cost is not fabricated in the economics.",
        "- Shield members have different economics and high lifetime value; the API surfaces a manual-review note rather than automatically excluding them.",
        "",
        "## Expected test submission",
        f"Generated `predictions.csv` contains {len(submission):,} test orders with a calibrated risk score in [0,1].",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {"cv_folds": fold_rows, "cv_auc_mean": cv_auc_mean, "cv_auc_std": cv_auc_std,
                "cv_ap_mean": cv_ap_mean, "cv_ap_std": cv_ap_std,
                "selected_threshold": best_call["threshold"], "best_accuracy_checked": best_accuracy,
                "prediction_rows": len(submission)}
    (ROOT / "reports" / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {PRED_PATH}")
    print(f"CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}; AP: {cv_ap_mean:.4f} ± {cv_ap_std:.4f}")
    print(f"Selected threshold: {best_call['threshold']:.3f}; test flags: {(test_score >= best_call['threshold']).sum():,}/{len(test_score):,}")
    print(f"Top features: {top_features}")


if __name__ == "__main__":
    main()
