from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test_unlabelled.csv"
CUSTOMERS_FILE = DATA_DIR / "customers.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"

# These are written during the return process or can reflect a return already being initiated.
LEAKY_COLS = ["pickup_scheduled_at", "last_service_event_type"]

# Fixed prior used only for smoothing historical rate features. It is not estimated from validation labels.
SMOOTHING_PRIOR = 0.11
SMOOTHING_ALPHA = 20.0

HIST_ENTITIES = ["sku", "family", "sales_channel", "payment_mode", "pincode_zone", "city"]
CATEGORICAL_COLS = [
    "customer_id", "sales_channel", "payment_mode", "family", "sku", "city", "state",
    "shield_member", "is_gift", "source", "pincode_zone"
]

TEXT_KEYWORDS = [
    "call", "leave", "gate", "address", "office", "security", "gift",
    "install", "urgent", "weekday", "door", "floor", "watch"
]

NUMERIC_COLS = [
    "customer_prior_orders", "customer_prior_returns", "prior_return_rate",
    "prior_orders_log", "prior_returns_log", "orders_per_year",
    "discount_pct", "discount_sq", "qty", "order_value_inr", "log_order_value",
    "unit_order_value", "value_vs_list", "effective_discount",
    "promised_delivery_days", "delivery_days_sq", "delivery_days_log",
    "delivery_pincode", "is_walkin", "days_since_launch", "customer_tenure_days",
    "order_hour", "order_dow", "order_month", "order_week", "order_is_weekend",
    "warranty_months", "list_price_inr", "is_cod", "is_emi", "is_shield", "is_gift_flag",
    "note_len", "note_words",
] + [f"note_{x}" for x in TEXT_KEYWORDS] + [f"{e}_hist_rate" for e in HIST_ENTITIES]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_data(base_dir: Path = DATA_DIR):
    train = pd.read_csv(base_dir / "train.csv", parse_dates=["order_placed_at"])
    test = pd.read_csv(base_dir / "test_unlabelled.csv", parse_dates=["order_placed_at"])
    customers = pd.read_csv(base_dir / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(base_dir / "products.csv", parse_dates=["launch_date"])
    return train, test, customers, products


def clean_orders(df: pd.DataFrame, products: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    d = df.copy()
    d["order_placed_at"] = pd.to_datetime(d["order_placed_at"], errors="coerce")
    if d["order_placed_at"].isna().any():
        raise ValueError("order_placed_at contains unparseable timestamps")

    if is_train:
        d["_source_rank"] = d["source"].map({"crm": 0, "partner_feed": 1}).fillna(9)
        d = (
            d.sort_values(["order_id", "_source_rank"])
             .drop_duplicates("order_id", keep="first")
             .drop(columns="_source_rank")
             .reset_index(drop=True)
        )

    # October 2025 festive orders were stored at exactly 100x the expected merchandise value.
    ref = products[["sku", "list_price_inr"]]
    tmp = d.merge(ref, on="sku", how="left")
    expected = tmp["list_price_inr"] * tmp["qty"] * (1 - tmp["discount_pct"] / 100.0)
    ratio = tmp["order_value_inr"] / expected.replace(0, np.nan)
    october = (tmp["order_placed_at"].dt.year == 2025) & (tmp["order_placed_at"].dt.month == 10)
    bad_gateway = october & ratio.between(99.999, 100.001)
    d.loc[bad_gateway.to_numpy(), "order_value_inr"] = d.loc[bad_gateway.to_numpy(), "order_value_inr"] / 100.0
    return d


def add_base_features(df: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["order_placed_at"] = pd.to_datetime(d["order_placed_at"], errors="coerce")
    if d["order_placed_at"].isna().any():
        raise ValueError("order_placed_at contains unparseable timestamps")
    d = d.merge(customers, on="customer_id", how="left", validate="many_to_one")
    d = d.merge(products, on="sku", how="left", validate="many_to_one")

    missing_customer = d["signup_date"].isna().sum()
    missing_product = d["launch_date"].isna().sum()
    if missing_customer or missing_product:
        raise ValueError(f"Reference join failed: {missing_customer} customer rows, {missing_product} product rows")

    # Post-return/process signals are intentionally excluded.
    d = d.drop(columns=[c for c in LEAKY_COLS if c in d.columns], errors="ignore")

    d["order_hour"] = d["order_placed_at"].dt.hour.astype(int)
    d["order_dow"] = d["order_placed_at"].dt.dayofweek.astype(int)
    d["order_month"] = d["order_placed_at"].dt.month.astype(int)
    d["order_week"] = d["order_placed_at"].dt.isocalendar().week.astype(int)
    d["order_is_weekend"] = (d["order_dow"] >= 5).astype(int)
    d["days_since_launch"] = (d["order_placed_at"] - d["launch_date"]).dt.days.clip(lower=0)
    d["customer_tenure_days"] = (d["order_placed_at"] - d["signup_date"]).dt.days.clip(lower=0)

    d["prior_return_rate"] = (
        d["customer_prior_returns"] / d["customer_prior_orders"].replace(0, np.nan)
    ).fillna(0.0)
    d["prior_orders_log"] = np.log1p(d["customer_prior_orders"])
    d["prior_returns_log"] = np.log1p(d["customer_prior_returns"])
    d["orders_per_year"] = (
        d["customer_prior_orders"] / (d["customer_tenure_days"].clip(lower=30) / 365.0)
    ).clip(upper=20)

    d["log_order_value"] = np.log1p(d["order_value_inr"].clip(lower=0))
    d["unit_order_value"] = d["order_value_inr"] / d["qty"].clip(lower=1)
    d["value_vs_list"] = (
        d["order_value_inr"] / (d["list_price_inr"] * d["qty"]).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    d["effective_discount"] = (1 - d["value_vs_list"]).clip(-5, 1)
    d["discount_sq"] = d["discount_pct"] ** 2

    d["is_walkin"] = (d["delivery_pincode"] == 0).astype(int)
    d["pincode_zone"] = (d["delivery_pincode"] // 10000).astype(int).astype(str)

    d["is_cod"] = (d["payment_mode"] == "cod").astype(int)
    d["is_emi"] = (d["payment_mode"] == "emi").astype(int)
    d["is_shield"] = (d["shield_member"] == "Y").astype(int)
    d["is_gift_flag"] = (d["is_gift"] == "Y").astype(int)

    d["delivery_days_sq"] = d["promised_delivery_days"] ** 2
    d["delivery_days_log"] = np.log1p(d["promised_delivery_days"])

    note = d["delivery_note"].fillna("").astype(str).str.lower()
    d["note_len"] = note.str.len()
    d["note_words"] = note.str.split().str.len()
    for kw in TEXT_KEYWORDS:
        d[f"note_{kw}"] = note.str.contains(kw, regex=False).astype(int)

    # CatBoost handles categorical values directly; normalize missing values to a stable token.
    for col in CATEGORICAL_COLS:
        d[col] = d[col].fillna("MISSING").astype(str)

    return d


def add_expanding_history_rates(train_features: pd.DataFrame, entities: Iterable[str] = HIST_ENTITIES) -> pd.DataFrame:
    """Create causal historical return rates for each training row using only earlier rows."""
    d = train_features.sort_values(["order_placed_at", "order_id"]).copy()
    for entity in entities:
        sums: dict[str, int] = {}
        counts: dict[str, int] = {}
        rates = []
        for key, target in zip(d[entity], d["returned"].astype(int)):
            s = sums.get(key, 0)
            c = counts.get(key, 0)
            rates.append((s + SMOOTHING_ALPHA * SMOOTHING_PRIOR) / (c + SMOOTHING_ALPHA))
            sums[key] = s + int(target)
            counts[key] = c + 1
        d[f"{entity}_hist_rate"] = rates
    return d.sort_index()


def add_fitted_history_rates(history_features: pd.DataFrame, apply_features: pd.DataFrame,
                             entities: Iterable[str] = HIST_ENTITIES) -> pd.DataFrame:
    """Apply target-history statistics fitted only on earlier training data."""
    out = apply_features.copy()
    for entity in entities:
        g = history_features.groupby(entity)["returned"].agg(["sum", "count"])
        mapping = g.to_dict("index")
        values = []
        for key in out[entity]:
            stat = mapping.get(key)
            if stat is None:
                values.append(SMOOTHING_PRIOR)
            else:
                values.append((stat["sum"] + SMOOTHING_ALPHA * SMOOTHING_PRIOR) /
                              (stat["count"] + SMOOTHING_ALPHA))
        out[f"{entity}_hist_rate"] = values
    return out


def make_train_features(train_df: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    base = add_base_features(train_df, customers, products)
    return add_expanding_history_rates(base)


def make_train_val_features(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            customers: pd.DataFrame, products: pd.DataFrame):
    hist = make_train_features(train_df, customers, products)
    val = add_base_features(val_df, customers, products)
    val = add_fitted_history_rates(hist, val)
    return hist, val
