import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


XTrain = pd.read_csv('data/processed/X_train.csv')
yTrain = pd.read_csv('data/processed/y_train.csv')


XTest = pd.read_csv('data/processed/X_test.csv')
yTest = pd.read_csv('data/processed/y_test.csv')


# Initalize
model = LinearRegression()


# Train
model.fit(XTrain, yTrain)


# Predict
yPred = model.predict(XTest)


# Metrics
mse = mean_squared_error(yTest, yPred)
mae = mean_absolute_error(yTest, yPred)
r2 = r2_score(yTest, yPred)

