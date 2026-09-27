from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from catboost import Pool

from src.features import DATA_DIR, CATEGORICAL_COLS, FEATURE_COLS, HIST_ENTITIES, add_base_features, add_fitted_history_rates

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
MODEL = None
ARTIFACT = None
HISTORY = None
CUSTOMERS = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
PRODUCTS = pd.read_csv(DATA_DIR / "products.csv", parse_dates=["launch_date"])


def load_model():
    global MODEL, ARTIFACT, HISTORY
    if not MODEL_PATH.exists():
        MODEL = ARTIFACT = None
        return
    ARTIFACT = joblib.load(MODEL_PATH)
    MODEL = ARTIFACT["model"]
    from src.features import clean_orders, make_train_features
    _raw_train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["order_placed_at"])
    _clean_train = clean_orders(_raw_train, PRODUCTS, is_train=True)
    HISTORY = make_train_features(_clean_train, CUSTOMERS, PRODUCTS)

load_model()

app = Flask(__name__)

HTML = """<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Kestrel Returns Risk</title>
<style>
body{font-family:Arial,sans-serif;background:#f6f7f9;margin:0;color:#17202a}.wrap{max-width:950px;margin:32px auto;padding:0 18px}.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 4px 14px rgba(0,0,0,.04)}h1{margin:0 0 6px}.muted{color:#64748b}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.field label{font-size:12px;color:#475569;display:block;margin-bottom:5px}.field input,.field select{width:100%;padding:9px;border:1px solid #cbd5e1;border-radius:8px;box-sizing:border-box}.btn{margin-top:16px;width:100%;padding:11px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer}.score{font-size:44px;font-weight:800}.badge{font-weight:700;padding:6px 11px;border-radius:18px;display:inline-block}.high{background:#fee2e2;color:#991b1b}.med{background:#fef3c7;color:#92400e}.low{background:#dcfce7;color:#166534}.reason{padding:10px 12px;border-left:4px solid #64748b;background:#f8fafc;border-radius:6px;margin:8px 0}.reason.up{border-left-color:#dc2626}.reason.down{border-left-color:#16a34a}.note{padding:11px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;color:#1e3a8a}.err{color:#991b1b;background:#fee2e2;padding:10px;border-radius:8px;display:none}.full{grid-column:1/-1}</style></head>
<body><div class='wrap'>
<h1>Kestrel Returns Risk</h1><div class='muted'>Pre-dispatch scoring demo. The service rejects post-return fields by design.</div>
<div id='missing' class='card' style='display:none'>Model not found. Run <code>python -m src.train</code> first.</div>
<div class='card'><h3>Order</h3><div class='grid'>
<div class='field'><label>Order placed at</label><input id='order_placed_at' value='2026-09-26 10:00'></div>
<div class='field'><label>Customer ID</label><input id='customer_id' value='KC105196'></div>
<div class='field'><label>SKU</label><input id='sku' value='KH-IC-03'></div>
<div class='field'><label>Sales channel</label><select id='sales_channel'><option>app</option><option>web</option><option>marketplace</option><option>partner_outlet</option></select></div>
<div class='field'><label>Payment mode</label><select id='payment_mode'><option>prepaid_upi</option><option>prepaid_card</option><option selected>cod</option><option>emi</option></select></div>
<div class='field'><label>Discount %</label><input id='discount_pct' type='number' value='20'></div>
<div class='field'><label>Qty</label><input id='qty' type='number' value='1'></div>
<div class='field'><label>Order value (₹)</label><input id='order_value_inr' type='number' value='9599.2'></div>
<div class='field'><label>Promised delivery days</label><input id='promised_delivery_days' type='number' value='7'></div>
<div class='field'><label>Delivery pincode</label><input id='delivery_pincode' type='number' value='400151'></div>
<div class='field'><label>Gift?</label><select id='is_gift'><option selected>N</option><option>Y</option></select></div>
<div class='field'><label>Prior orders</label><input id='customer_prior_orders' type='number' value='2'></div>
<div class='field'><label>Prior returns</label><input id='customer_prior_returns' type='number' value='1'></div>
<div class='field'><label>Delivery note</label><input id='delivery_note' value='Leave with security'></div>
<div class='field'><label>Source</label><select id='source'><option selected>crm</option><option>partner_feed</option></select></div>
</div><button class='btn' onclick='predict()'>Check return risk</button><div id='err' class='err'></div></div>
<div id='result' class='card' style='display:none'><h3>Assessment</h3><div class='score' id='score'>—</div><span id='risk' class='badge'>—</span><p><b>Recommendation:</b> <span id='rec'>—</span></p><div id='shield'></div><h4>Reasons</h4><div id='reasons'></div></div>
</div><script>
const fields=['order_placed_at','customer_id','sku','sales_channel','payment_mode','discount_pct','qty','order_value_inr','promised_delivery_days','delivery_pincode','is_gift','customer_prior_orders','customer_prior_returns','delivery_note','source'];
function val(id){const e=document.getElementById(id);return e.type==='number'?Number(e.value):e.value}
async function predict(){document.getElementById('err').style.display='none'; const p={};fields.forEach(k=>p[k]=val(k));try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const d=await r.json();if(!r.ok)throw new Error(d.error||'Prediction failed');document.getElementById('result').style.display='block';document.getElementById('score').textContent=(100*d.score).toFixed(1)+'%';const b=document.getElementById('risk');b.textContent=d.risk_level;b.className='badge '+(d.risk_level==='HIGH'?'high':d.risk_level==='MEDIUM'?'med':'low');document.getElementById('rec').textContent=d.recommendation;document.getElementById('shield').innerHTML=d.shield_note?'<div class="note">'+d.shield_note+'</div>':'';document.getElementById('reasons').innerHTML=d.reasons.map(x=>'<div class="reason '+x.direction+'">'+x.text+'</div>').join('')}catch(e){const er=document.getElementById('err');er.textContent=e.message;er.style.display='block'}}
</script></body></html>"""


def _record_from_json(payload):
    required = [
        "order_placed_at", "customer_id", "sku", "sales_channel", "payment_mode", "discount_pct",
        "qty", "order_value_inr", "promised_delivery_days", "delivery_pincode", "is_gift",
        "customer_prior_orders", "customer_prior_returns", "delivery_note", "source"
    ]
    missing = [x for x in required if x not in payload]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    # Refuse post-return/process fields even if a client sends them.
    forbidden = [c for c in payload if c in {"pickup_scheduled_at", "last_service_event_type", "returned"}]
    if forbidden:
        raise ValueError("Post-return/outcome fields are not accepted: " + ", ".join(forbidden))
    return pd.DataFrame([payload])


def _reason_text(feature, row):
    mapping = {
        "customer_prior_returns": f"Customer has {int(row['customer_prior_returns'])} prior return(s).",
        "prior_return_rate": f"Customer prior-return rate is {row['prior_return_rate']:.0%}.",
        "sku_hist_rate": f"This SKU's observed historical return rate is {row['sku_hist_rate']:.0%} before this order.",
        "family": f"Product family is {row['family'] }.",
        "family_hist_rate": f"Historical return rate for {row['family']} before this order is {row['family_hist_rate']:.0%}.",
        "sales_channel": f"Sales channel is {row['sales_channel'] }.",
        "sales_channel_hist_rate": f"Historical return rate for {row['sales_channel']} before this order is {row['sales_channel_hist_rate']:.0%}.",
        "payment_mode": f"Payment mode is {row['payment_mode'] }.",
        "payment_mode_hist_rate": f"Historical return rate for {row['payment_mode']} before this order is {row['payment_mode_hist_rate']:.0%}.",
        "is_cod": "Cash on Delivery is associated with higher historical return risk.",
        "is_shield": "Shield membership is associated with higher return frequency; treat this as a manual-review consideration.",
        "promised_delivery_days": f"Promised delivery is {int(row['promised_delivery_days'])} days.",
        "effective_discount": f"Effective discount signal is {row['effective_discount']:.0%}.",
        "days_since_launch": f"Product age at order is {int(row['days_since_launch'])} days.",
        "is_walkin": "No address was captured at checkout (walk-in/default pincode).",
    }
    return mapping.get(feature, feature.replace('_',' '))


def predict_one(payload):
    if MODEL is None:
        raise RuntimeError("Model not found. Run 'python -m src.train' first.")
    row_raw = _record_from_json(payload)
    base = add_base_features(row_raw, CUSTOMERS, PRODUCTS)
    # For a single live record, history statistics come from all labeled training orders.
    row = add_fitted_history_rates(HISTORY, base, HIST_ENTITIES)
    X = row[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].fillna("MISSING").astype(str)
    raw_score = float(MODEL.predict_proba(X)[:,1][0])
    score = float(ARTIFACT["calibrator"].predict([raw_score])[0])

    call_threshold = float(ARTIFACT["call_threshold"])
    hold_threshold = float(ARTIFACT.get("hold_threshold", 0.40))
    if score >= hold_threshold:
        risk = "HIGH"
        recommendation = "Hold for manual review and confirmation call"
    elif score >= call_threshold:
        risk = "MEDIUM"
        recommendation = "Confirmation call / manual review"
    else:
        risk = "LOW"
        recommendation = "Dispatch normally"

    # Per-row SHAP contributions from the CatBoost model.
    pool = Pool(X, cat_features=[X.columns.get_loc(c) for c in CATEGORICAL_COLS])
    shap = MODEL.get_feature_importance(pool, type="ShapValues")[0][:-1]
    pairs = list(zip(FEATURE_COLS, shap))
    top_up = sorted([x for x in pairs if x[1] > 0], key=lambda z: -z[1])[:3]
    top_down = sorted([x for x in pairs if x[1] < 0], key=lambda z: z[1])[:1]
    reasons = [{"direction":"up","text":_reason_text(f,row.iloc[0])} for f,_ in top_up]
    reasons += [{"direction":"down","text":_reason_text(f,row.iloc[0])} for f,_ in top_down]
    if not reasons:
        reasons=[{"direction":"up","text":"No single feature dominates this prediction; treat the score as the model's combined risk estimate."}]
    shield_note = None
    if str(payload.get("customer_id")) in set(CUSTOMERS.loc[CUSTOMERS.shield_member=='Y','customer_id']):
        shield_note = "Shield member: higher observed return frequency but high customer value. Keep a human in the decision loop."
    return {"score": round(score,4), "risk_level": risk, "recommendation": recommendation,
            "reasons": reasons[:4], "shield_note": shield_note}

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({"status":"ok", "model_loaded": MODEL is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error":"JSON body must be a single object/record"}), 400
        return jsonify(predict_one(payload))
    except RuntimeError as e:
        return jsonify({"error":str(e)}), 503
    except ValueError as e:
        return jsonify({"error":str(e)}), 400
    except Exception as e:
        return jsonify({"error":f"Prediction failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
