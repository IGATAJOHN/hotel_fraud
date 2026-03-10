import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    df["occupancy_lag_1"] = df["occupancy"].shift(1)
    df["occupancy_lag_7"] = df["occupancy"].shift(7)
    df["adr_lag_7"] = df["adr"].shift(7)

    # Rolling features
    df["occ_roll_3"] = df["occupancy"].rolling(3).mean()
    df["occ_roll_7"] = df["occupancy"].rolling(7).mean()
    df["adr_roll_7"] = df["adr"].rolling(7).mean()
    df["occ_std_7"] = df["occupancy"].rolling(7).std()

    return df

def prepare_demand_features(history: list) -> pd.DataFrame:
    # Handle both dicts and Pydantic models by using dict() fallback
    data = [h.dict() if hasattr(h, 'dict') else dict(h) for h in history]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    
    df = compute_features(df)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Not enough historical data to compute features")

    return df
