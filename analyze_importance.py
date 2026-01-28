import joblib
import pandas as pd

def show_importance():
    try:
        model = joblib.load("model.pkl")
        FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
        
        # Extract importance
        importances = model.get_feature_importance()
        
        df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        print("--- Top 10 Features ---")
        print(df.head(10).to_string(index=False))
        
        print("\n--- Summary ---")
        cumulative = df['importance'].cumsum()
        n_90 = (cumulative <= 90).sum() + 1
        print(f"Number of features contributing to 90% of the prediction: {n_90} out of {len(FEATURE_COLUMNS)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    show_importance()
