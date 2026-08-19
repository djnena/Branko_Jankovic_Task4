import joblib
import pandas as pd
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/car_price_model.joblib"

#ucitavanje skupa podataka
print("Loading dataset...")
 
df = pd.read_csv(DATA_PATH)

#delimo podatke na ulazne i izlazne
print("Splitting features and target...")
 
X, y = split_features_and_target(df)

print("Splitting data into training and test sets...")
 
#delimo ulazne i izlazne podatke za testiranje i treniranje
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#kreiranje modela
print("Creating model pipeline...")
 
model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ]
)

#treniranje modela
print("Training model...")
 
model.fit(X_train, y_train)

#cuvanje modela
print("Saving model...")
 
joblib.dump(model, MODEL_PATH)
 
print(f"Model saved to: {MODEL_PATH}")