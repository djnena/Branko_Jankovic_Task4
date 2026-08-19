import re
import pandas as pd

RAW_DATA_PATH = "Branko_Jankovic_Task4/data/cars.csv"
CLEANED_DATA_PATH = "Branko_Jankovic_Task4/data/cars_cleaned.csv"
MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}

#funkcija koja sredjuje nazive kolona: mala slova, bez specijalnih karaktera i razmaka, sa donjom
#crtom izmedju
def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    new_columns = []
 
    for col in df.columns:
        clean_col = col.strip().lower()
 
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
 
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
 
        new_columns.append(clean_col)
 
    df.columns = new_columns

    return df

#funkcija uklanja nedostajuce vrednosti iz liste MISSING_LIKE_VALUES sa NAN
def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df

#funkcija koja sredjuje kategorijske vrednosti, sva slova u kolonama se pretvaraju u mala
def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame: 
      df = df.copy() 
      
      categorical_columns = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]
      
      for col in categorical_columns: 
            if col in df.columns: 
                  df[col] = ( 
                  df[col] 
                  .astype("string") 
                  .str.strip() 
                  .str.lower() 
                  ) 
                 
      return df


#poziv svih funkcija
def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_replace_missing_like_values)
        .pipe(_clean_categorical_values)
        .reset_index(drop=True)
    )
 
    return df_clean

#poziv skripte iz terminala
def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")
 
    df_raw = pd.read_csv(RAW_DATA_PATH)
 
    print("Cleaning dataset...")
 
    df_cleaned = clean(df_raw)
 
    print("Saving cleaned dataset...")
 
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
 
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    
#Ako se fajl pokreće direktno iz terminala, pozovi main() funkciju
#ako se inportuje kod u nekoj drugoj skripti ovo se nece pozvati
if __name__ == "__main__":
    main()