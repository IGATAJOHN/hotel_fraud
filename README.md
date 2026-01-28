# Hotel Fraud Detection API

A robust FastAPI-based prediction service for hotel booking cancellations and fraud detection using CatBoost and SHAP explanations.

## Features
- **FastAPI Endpoint**: Score bookings via POST requests.
- **CatBoost Model**: Optimized for high-accuracy fraud detection.
- **SHAP Explanations**: Identify the top 5 reasons for every prediction.
- **Robust Schema Handling**: Automatically handles missing variables and name mapping.

## Setup

### Prerequisites
- Python 3.8+
- [CatBoost](https://catboost.ai/)
- [SHAP](https://shap.readthedocs.io/)
- [Joblib](https://joblib.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)

### Installation
```bash
pip install fastapi uvicorn pandas numpy catboost shap joblib
```

### Running the API
```bash
uvicorn main:app --reload
```

## API Usage

### Score Booking
**Endpoint**: `POST /score`

**Sample Request**:
```json
{
  "data": {
    "lead_time": 1,
    "room_price": 150,
    "stay_length": 3,
    "is_new_customer": 1
  }
}
```

**Sample Response**:
```json
{
  "fraud_probability": 0.5106,
  "flagged_for_review": true,
  "top_reasons": ["revpar", "amount", "adr", "stay_length", "lead_time_days"]
}
```
