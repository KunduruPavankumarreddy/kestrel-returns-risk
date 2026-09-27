# Evidence — Kestrel Returns Risk

## Dataset and validation
- Training rows after partner re-import deduplication: 10,504 unique orders.
- Historical return rate: 11.42% (1,200 returns).
- Temporal validation: five expanding-window monthly folds on 2026-02, 2026-03, 2026-04, 2026-05, 2026-06.
- Target-derived historical-rate features are recomputed inside each fold using earlier training data only.
- `pickup_scheduled_at` and `last_service_event_type` are excluded because the operations policy says they are written during the return process.

## Cross-validation
| Validation month | Orders | Returns | ROC-AUC | Average Precision | Best iterations |
|---|---:|---:|---:|---:|---:|
| 2026-02 | 664 | 85 | 0.806 | 0.505 | 124 |
| 2026-03 | 743 | 81 | 0.764 | 0.287 | 241 |
| 2026-04 | 705 | 85 | 0.778 | 0.427 | 90 |
| 2026-05 | 722 | 83 | 0.783 | 0.392 | 111 |
| 2026-06 | 699 | 77 | 0.775 | 0.419 | 94 |
| **Mean** | | | **0.781 ± 0.014** | **0.406 ± 0.070** | |

### Pre-submission expectation
Expected held-out ranking performance: ROC-AUC ≈ **0.781** and average precision ≈ **0.406**, based only on the five temporal validation folds.

## Threshold analysis (out-of-fold, calibrated scores)
A confirmation call costs ₹45 and the pilot reported preventing about 35% of returns on called orders. The policy gives a ₹1,150 average return cost. The table therefore uses `TP × 35% × ₹1,150 − flags × ₹45` as an indicative call-pilot value. It does **not** invent a cancellation loss from order value.

| Threshold | Accuracy | Precision | Recall | Flags | Indicative call-pilot value / validation sample |
|---:|---:|---:|---:|---:|---:|
| 0.100 | 0.596 | 0.200 | 0.825 | 1,693 | ₹60,262 |
| 0.125 | 0.754 | 0.269 | 0.650 | 992 | ₹62,827 |
| 0.150 | 0.792 | 0.299 | 0.584 | 804 | ₹60,420 |
| 0.200 | 0.854 | 0.391 | 0.455 | 478 | ₹53,758 |
| 0.250 | 0.857 | 0.397 | 0.448 | 463 | ₹53,225 |
| 0.300 | 0.882 | 0.486 | 0.299 | 253 | ₹38,122 |
| 0.350 | 0.883 | 0.496 | 0.287 | 238 | ₹36,785 |
| 0.400 | 0.893 | 0.660 | 0.161 | 100 | ₹22,065 |
| 0.500 | 0.893 | 0.660 | 0.161 | 100 | ₹22,065 |

**Selected threshold:** 0.125, because it has the highest indicative call-pilot value on the pooled out-of-fold sample. This is an operating threshold for a call-first pilot, not a claim that every flagged order should automatically be held.
**Highest observed OOF accuracy among checked thresholds:** 0.893 at threshold 0.400. This is reported with recall/precision because 88.6% of historical orders were non-returns.
OOF Brier score after isotonic calibration: 0.0856.

## Data-quality decisions
- 651 partner-feed re-imports were deduplicated by `order_id`, keeping the CRM copy.
- October 2025 order values with an exact 100× merchandise-value ratio were divided by 100, matching the payment-gateway issue noted in the email thread.
- `delivery_pincode = 000000` is retained as an explicit walk-in/no-address signal, per the data pack.

## Failure modes / limitations
- Returns are only ~11% of orders, so accuracy alone overstates usefulness.
- The model cannot see every real-world reason for return; text is represented only through simple deterministic note signals.
- The policy gives a 12% cancellation rate for holds over 24 hours but does not specify the financial loss per cancellation, so that cost is not fabricated in the economics.
- Shield members have different economics and high lifetime value; the API surfaces a manual-review note rather than automatically excluding them.

## Expected test submission
Generated `predictions.csv` contains 2,096 test orders with a calibrated risk score in [0,1].