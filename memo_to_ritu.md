# Memo to Ritu Deshpande — Kestrel Home

**From:** Analytics team  
**Re:** Returns risk model — pre-dispatch pilot  
**Date:** 26 September 2026

## The decision

Do not auto-hold every order flagged by the model yet. Start a one-week **confirmation-call pilot** on orders at or above a calibrated risk score of **0.125** and keep a stricter **0.40** tier for manual hold/review.

This keeps the model in the dispatch workflow while testing the operational intervention that Kestrel has already piloted, without assuming a financial loss for a cancelled held order that the policy does not quantify.

## The number

We evaluated the model on five future-month validation windows (February–June 2026), always training on earlier orders only. The model's mean **ROC-AUC was 0.781 ± 0.014** and mean Average Precision was **0.406 ± 0.070**.

The historical return rate was 11.4%, so accuracy is not the right headline by itself. The highest accuracy among the checked operating thresholds was **89.3%**, at a threshold of 0.40, where recall falls to about 16%. The model should therefore be treated as a ranking/risk system, not as a claim that the board's 95% accuracy target has been met.

At the pilot threshold of 0.125, the out-of-fold data flags about **28% of orders**, with **65% recall** and **27% precision**. On the current unlabelled snapshot of 2,096 orders, the same threshold flags **590 orders**.

## The rupees

Kestrel's Operations Policy uses **₹1,150 per returned order** and **₹45 per completed confirmation call**. The spring pilot reported that a confirmation call prevented about **35% of returns that otherwise would have happened on called orders**.

Using those stated assumptions, the threshold-0.125 validation cohort produces an indicative call-pilot value of about **₹62.8k across 3,533 validation orders**, or approximately **₹1.78 lakh per 10,000 orders** when scaled linearly. This is a pilot estimate, not a booked saving.

We did not subtract a made-up cancellation loss from held orders: the policy gives a 12% cancellation rate for holds exceeding 24 hours, but it does not state the rupee loss per cancellation.

## What changed in the data

We removed 651 partner-feed re-import duplicates, correcting the training set to 10,504 unique historical orders. We also corrected the October 2025 payment-gateway anomaly where affected stored order values were exactly 100× the merchandise value implied by price, quantity and discount.

We excluded `pickup_scheduled_at` and `last_service_event_type`. The Operations Policy states that reverse pickup is booked and the service system records `REVERSE_PICKUP` once a return has started; using these fields would leak the outcome into a pre-dispatch prediction.

## What she should do next week

1. Run the model in shadow mode for a few days and compare scores with actual outcomes as they arrive.
2. Start the confirmation-call pilot on the 0.125+ band; record whether the call changed the outcome and record any operational reasons.
3. Keep the 0.40+ band for manual review/hold rather than automatic blocking, especially for Shield members.
4. Review the result after one week using actual return rate, recall, call volume, cancellation/complaint rate, and realised rupee impact. If those numbers support it, move the threshold and hold policy from pilot to a controlled production rule.
