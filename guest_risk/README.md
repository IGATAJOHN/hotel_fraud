# Guest Risk Profiling

**Booking-level fraud** answers: *"Is this booking fraudulent?"*  
**Guest risk profiling** answers: *"Is this guest risky over time?"*

This module aggregates guest behavior and trains a **guest-level risk model**. It consumes fraud outputs and historical behavior; it does not replace the existing fraud model.

---

## 1. Feature table (guest-level)

All features are **aggregates per guest** (derived from bookings, fraud scores, and behavior).

| Feature | Type | Derivation |
|--------|------|------------|
| `guest_id` | str | Identifier (excluded from model input) |
| `total_bookings` | int | Count of bookings |
| `cancellation_rate` | float | Cancellations / total bookings |
| `no_show_rate` | float | No-shows / total bookings |
| `refund_request_rate` | float | Refund requests / total bookings |
| `avg_spend` | float | Mean booking value |
| `avg_lead_time` | float | Mean days between book and check-in |
| `avg_num_guests` | float | Mean party size |
| `fraud_booking_ratio` | float | Bookings flagged fraud / total (from fraud model) |
| `fraud_booking_count` | int | Number of bookings flagged as fraud |
| `mean_payment_declines` | float | Mean payment decline rate across bookings |
| `chargeback_ever` | int | 1 if guest ever had a chargeback |
| `tenure_days` | float | Days since first booking |

**Model features** (what the model sees): all of the above except `guest_id`.  
**Target**: `guest_is_risky` (see label logic below).

---

## 2. Label logic (business-defined)

`guest_is_risky = 1` if **any** of the following is true:

- **Fraud count:** ≥ 2 bookings flagged as fraudulent, OR  
- **Fraud ratio:** ≥ 30% of their bookings flagged as fraud, OR  
- **Chargeback:** guest has at least one chargeback

Otherwise `guest_is_risky = 0`.

This is defined in code in `schema.py` and used by the feature builder and synthetic generator.

---

## 3. Output (scoring)

The trained pipeline returns per-guest risk, e.g.:

```json
{
  "guest_id": "GUEST_1029",
  "risk_score": 0.81,
  "risk_tier": "HIGH",
  "confidence": 0.92,
  "top_risk_factors": ["high_refund_rate", "frequent_no_shows", "prior_fraud_bookings"]
}
```

---

## 4. Files (scope)

```
guest_risk/
├── README.md              (this file)
├── schema.py              (feature list, label rules, data contract)
├── generate_synthetic.py  (synthetic training data)
├── feature_builder.py     (aggregate bookings → guest features) [Step 2]
├── train_guest_risk.py    (train + save .pkl)                  [Step 2]
├── score_guest.py         (score one or many guests)            [Step 2]
└── guest_risk_model.pkl   (after training)
```

---

## 5. How this fits

```
Booking → Fraud Model (existing)
          ↓
     Guest Aggregator (this module)
          ↓
     Guest Risk Model (this module)
```

We consume fraud outputs; we do not replace them.
