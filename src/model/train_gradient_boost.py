import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# Load data
XTrain = pd.read_csv('data/processed/X_train.csv')
yTrain = pd.read_csv('data/processed/y_train.csv')
XTest = pd.read_csv('data/processed/X_test.csv')
yTest = pd.read_csv('data/processed/y_test.csv')


# Initialize the model
model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)


# Train
model.fit(XTrain, yTrain)


# Predict
yPred = model.predict(XTest)


# Metrics
mse = mean_squared_error(yTest, yPred)
mae = mean_absolute_error(yTest, yPred)
r2 = r2_score(yTest, yPred)


print(mse)
print(mae)
print(r2)