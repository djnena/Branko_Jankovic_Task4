import pandas as pd
import numpy as np



#racunanje koliko je automobil star
def _create_car_age_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df["car_age"]=(pd.Timestamp.today().year - df["year"])
 
    return df

#racunanje koliko je kilometara predjeno godisnje
def _create_mileage_per_year_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df["mileage_per_year"]=df["mileage_kilometers"]/(pd.Timestamp.today().year - df["year"]).replace(0, np.nan)
 
    return df

#racunanje zapremine motora iz cm3 u litre
def _create_engine_volume_liters_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df["engine_volume_liters"] = df["volume_cm3"] / 1000
 
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
 
    df_features = (
        df
        .pipe(_create_car_age_columns)
        .pipe(_create_mileage_per_year_columns)
        .pipe(_create_engine_volume_liters_columns)
        .reset_index(drop=True)
    )
 
    return df_features

CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_cleaned_with_features.csv"

def main() -> None:
    """Load cleaned data, build features, and save the feature-engineered dataset."""
    print("Loading cleaned dataset...")
 
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
 
    print("Building features...")
 
    df_features = build_features(df_cleaned)
 
    print("Saving feature-engineered dataset...")
 
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
 
    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")
    
if __name__ == "__main__":
      main()