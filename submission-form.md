# Submission Form — Kestrel Home Returns Risk (Variant A)

## What I built

A pre-dispatch returns-risk scoring service that predicts the probability an order will be returned. Delivered as:

1. `predictions.csv` — one calibrated score for each of the 2,096 unlabelled test orders.
2. Flask API with `POST /predict`, plus a single-page browser UI.
3. `evidence.md` — leakage-safe temporal validation, threshold analysis, data-quality decisions and failure modes.
4. `memo_to_ritu.md` — one-page business memo.
5. This completed submission form.

## Model

**Algorithm:** CatBoostClassifier with isotonic calibration from out-of-fold predictions.

**Validation:** Five expanding-window monthly temporal folds. Each fold trains on earlier orders and validates on a later month.

**Expected test ROC-AUC (written before hidden test labels):** **0.781**.

**Expected test Average Precision:** **0.406**.

Rationale: these are the means of the five temporal validation folds covering February–June 2026. No hidden test outcomes were used.

### Operating thresholds

- **0.125+**: confirmation-call / manual-review pilot band. This threshold had the highest indicative call-pilot value in the pooled out-of-fold sample.
- **0.40+**: stricter manual hold/review band.

At 0.125 on out-of-fold predictions: accuracy **75.4%**, precision **26.9%**, recall **65.0%**. This intentionally does not claim 95% accuracy; the model is primarily a risk-ranking system and the historical non-return rate is 88.6%.

The highest accuracy among the checked thresholds was **89.3%** at threshold 0.40, where recall was only 16.1%.

## Key decisions made

### 1. Removed post-return leakage

Dropped `pickup_scheduled_at` and `last_service_event_type`. The policy says reverse pickup is booked and `REVERSE_PICKUP` is recorded after a return has started. These fields therefore cannot be used for a genuine pre-dispatch prediction.

### 2. Deduplicated partner re-imports

651 `order_id` values appeared twice because partner-outlet orders were re-imported from the partner feed. Kept the CRM row as the canonical copy.

### 3. Corrected the October payment-gateway anomaly

For October 2025, 748 rows had an exact 100× relationship between stored order value and the merchandise value implied by list price, quantity and discount. Those stored values were divided by 100 before modelling.

### 4. Used causal target-history features

Historical return-rate features for SKU, product family, channel, payment mode, pincode zone and city are calculated using earlier observations only. Within each validation fold they are fitted on the training period and then applied to the future validation month.

### 5. Reframed the 95% accuracy requirement

The leakage-safe validation does not support a 95% accuracy claim. Rather than use post-return signals or training-set threshold metrics to produce a misleading number, I report ROC-AUC, Average Precision, precision, recall, accuracy and operational coverage together.

### 6. Shield handling

Shield membership is retained as a predictive feature, but the UI surfaces a manual-review note for Shield customers. The model does not automatically exclude or penalise them operationally because the policy describes Shield customers as a high-lifetime-value segment.

### 7. Financial calculation

The operations policy states: ₹1,150 average cost per returned order, ₹45 per confirmation call, and approximately 35% prevention of otherwise-expected returns on called orders. The financial analysis therefore uses `true returns flagged × 35% × ₹1,150 − calls × ₹45` as an indicative pilot value. It does not invent a rupee loss for the policy's 12% hold-cancellation rate because no cancellation-loss figure is supplied.

## AI tools used

- Claude was used for code generation/review and initial project analysis.
- ChatGPT was used for review, leakage audit, model-pipeline correction, validation design, and packaging.
- No paid API key or external model API is required by the delivered product.
- Model inference runs locally.

## What I discarded

- `pickup_scheduled_at` and `REVERSE_PICKUP` / `last_service_event_type` signals because they reveal the return process.
- Training-set threshold metrics as evidence of performance.
- Full-dataset target-derived rates inside temporal validation, because they can leak future outcomes into validation features.
- A cancellation-loss calculation based on order value, because the policy does not state that order value equals financial loss when a hold is cancelled.
- Random train/test validation as the primary evaluation method.

## Known limitations

- The historical return rate is only about 11.4%, so accuracy alone is a weak measure of usefulness.
- The model's weakest monthly validation fold was March 2026 (AUC 0.764); performance varies by time period.
- Delivery notes are represented with simple deterministic text signals rather than a language model.
- The model estimates risk; the operating intervention still needs a controlled pilot.
- The current unlabelled snapshot begins July 2026 and is scored using a model trained only on historical labelled orders.

## Working service

The project starts locally with:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python app.py
```

Then open `http://localhost:5050`.

`POST /predict` accepts a single dispatch-time order record and returns a calibrated score, risk level, recommendation and human-readable reasons. Post-return fields are rejected by the endpoint.
