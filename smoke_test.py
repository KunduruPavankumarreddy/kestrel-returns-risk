import json
import pandas as pd
from app import app

payload = {
    "order_placed_at": "2026-09-26 10:00",
    "customer_id": "KC105196",
    "sku": "KH-IC-03",
    "sales_channel": "web",
    "payment_mode": "cod",
    "discount_pct": 20,
    "qty": 1,
    "order_value_inr": 9599.20,
    "promised_delivery_days": 7,
    "delivery_pincode": 400151,
    "is_gift": "N",
    "customer_prior_orders": 2,
    "customer_prior_returns": 1,
    "delivery_note": "Leave with security",
    "source": "crm",
}

with app.test_client() as client:
    health = client.get('/health')
    print('health:', health.status_code, health.json)
    response = client.post('/predict', json=payload)
    print('predict:', response.status_code)
    print(json.dumps(response.json, indent=2))
    assert response.status_code == 200
    assert 0.0 <= response.json['score'] <= 1.0
