# Kestrel Home — Returns Risk (Variant A)

A reproducible pre-dispatch returns risk scorer. It predicts a 0–1 return-risk score for each order and exposes a single-record API plus a simple browser UI.

## Important submission choices

- Partner re-imports are deduplicated by `order_id`, keeping the CRM copy.
- October 2025 orders with an exact 100× payment-gateway value anomaly are corrected by dividing the affected stored value by 100.
- `pickup_scheduled_at` and `last_service_event_type` are excluded because the policy says they are written during the return process; using them would leak the outcome into the pre-dispatch prediction.
- Validation is expanding-window temporal CV, not random CV.
- Historical target-rate features are recomputed inside each validation fold using earlier training data only.
- Scores are isotonic-calibrated from out-of-fold predictions.
- The API accepts a dispatch-time record only and rejects outcome/post-return fields.

## Run on a clean machine

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
python app.py
```

Open http://localhost:5050

The endpoint is `POST /predict`. `GET /health` reports whether the trained model is loaded.

## Example API request

```json
{
  "order_placed_at": "2026-09-26 10:00",
  "customer_id": "KC105196",
  "sku": "KH-IC-03",
  "sales_channel": "web",
  "payment_mode": "cod",
  "discount_pct": 20,
  "qty": 1,
  "order_value_inr": 9599.2,
  "promised_delivery_days": 7,
  "delivery_pincode": 400151,
  "is_gift": "N",
  "customer_prior_orders": 2,
  "customer_prior_returns": 1,
  "delivery_note": "Leave with security",
  "source": "crm"
}
```

The API returns the calibrated score, a risk level, a recommendation, and human-readable reasons derived from the model's per-row feature contributions. It does not use an LLM or paid API.

## Validation headline

See `reports/evidence.md` for the exact fold metrics, threshold table, data-quality decisions, and limitations. The headline expected test metric is the mean temporal-validation ROC-AUC; no hidden test labels were used.

## Files

- `predictions.csv` — required test submission
- `model.pkl` — local trained model + calibrator
- `reports/evidence.md` — validation evidence
- `memo_to_ritu.md` — one-page business memo
- `submission-form.md` — completed submission form
- `screen_recording_script.md` — <=3 minute walkthrough script
- `docs/` — private client reference material; do not publish

## Data privacy

The supplied Kestrel data is client data. Keep this repository private and do not publish the CSVs or policy document.


