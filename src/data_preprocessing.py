import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

#ciljna promenljiva
TARGET_COLUMN = "priceusd"

#numericke karakteristike
NUMERIC_FEATURES = [
    "year",
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
]

#kategorijske kolone, ne mogu da se posloze, nemaju raspored
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit"
]

#ordinalne karakteristike
ORDINAL_FEATURES = [
    "segment",
]


#pomoćna funkcija vraća sve ulazne kolone koje će model koristiti
def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES

#razdvajanje ulaznih karakteristika i ciljne promenljive
def split_features_and_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
 
    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()
 
    return X, y

#Pretprocesiranje numeričkih kolona

def _build_numeric_transformer() -> Pipeline:
 
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
 
    return numeric_transformer

#za kategorijske vrednosti

def _build_categorical_transformer() -> Pipeline:
 
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
 
    return categorical_transformer

# za ordinalne kolone
#OrdinalEncoder ["low", "medium", "high", "jam"] ovo govori koji redosled katergorija treba da bude 0,1,2,3
def _build_ordinal_transformer() -> Pipeline:
 
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[
                    ["a", "b", "c", "d", "e", "f", "j", "s", "m"]
                ]
            )),
        ]
    )
 
    return ordinal_transformer

#za spajanje svega ovoga iznad 
#prvo obradi numericke kolone, zatim kategorijske kolone i na kraju ordinalne kolone
def build_preprocessor() -> ColumnTransformer:
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_transformer(), ORDINAL_FEATURES),
        ],
        remainder="drop"
    )
 
    return preprocessor