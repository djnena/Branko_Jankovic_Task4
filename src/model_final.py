import pandas as pd
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)

DATA_PATH = "data/cars_cleaned_with_features.csv"

df = pd.read_csv(DATA_PATH)

X, y = split_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Pipeline(
      steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", RandomForestRegressor())
        ]
    )
 
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
 
 
metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})
 
print("\nRandom Forest:")
print(metrics)
   