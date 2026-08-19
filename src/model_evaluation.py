import joblib
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/car_price_model.joblib"

print("Loading dataset...")
 
df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")
 
X, y = split_features_and_target(df)

print("Creating the same train/test split...")
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

test_size=0.2
random_state=42

print("Loading trained model...")
 
model = joblib.load(MODEL_PATH)

#pravljenje predikcije
print("Making predictions...")
 
y_pred = model.predict(X_test)

#mozemo prikazati prvih nekoliko predikcija
print(y_pred[:10])

#racunanje metrika regresije
print("Calculating regression metrics...")
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})
 
print("\nRegression metrics:")
print(metrics)

print("\nCreating prediction analysis table...")

prediction_analysis = pd.DataFrame({
      "actual_priceusd" : y_test.values,
      "predicted_priceusd" : y_pred,
})

#kreiranje tabele za analizu predikcije
prediction_analysis["error_min"] = (
    prediction_analysis["actual_priceusd"]
    - prediction_analysis["predicted_priceusd"]
)

#racunanje apsolutne greske
prediction_analysis["absolute_error_min"] = (
    prediction_analysis["error_min"].abs()
)

#prikaz nekoliko nasumicnih primera
print("\nPrediction examples:")
 
print(
    prediction_analysis
    .sample(10, random_state=42)
)

#primere na kojima je model najvise promasio
print("\nLargest prediction errors:")
 
print(
    prediction_analysis
    .sort_values("absolute_error_min", ascending=False)
    .head(10)
)