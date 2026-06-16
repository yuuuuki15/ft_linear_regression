import pandas as pd
import numpy as np

def load_model():
    try:
        model = pd.read_csv('model/default.csv')
        return model['theta0'][0], model['theta1'][0]
    except FileNotFoundError:
        return 0, 0

def estimate_price(mileage, theta0, theta1):
    price = mileage * theta1 + theta0
    return price

def mae(predicted, actual, m):
    return np.sum(np.abs(predicted - actual)) / m

def rmse(predicted, actual, m):
    return np.sqrt(np.sum((predicted - actual) ** 2) / m)

def r2(predicted, actual):
    avr = np.mean(actual)
    left = np.sum((predicted - actual) ** 2)
    right = np.sum((avr - actual) ** 2)
    return 1 - (left / right)

data = pd.read_csv('data.csv')
theta0, theta1 = load_model()
x = data['km']
y = data['price']
m = data.shape[0]
predicted_data = estimate_price(x, theta0, theta1)

mae_result = mae(predicted_data, y, m)
rmse_result = rmse(predicted_data, y, m)
r2_result = r2(predicted_data, y)

print(mae_result)
print(rmse_result)
print(r2_result)