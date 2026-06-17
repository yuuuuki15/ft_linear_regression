import pandas as pd
import numpy as np
import os

def min_max_scaling(mileage):
    min_val = mileage.min()
    max = mileage.max()
    diff = max - min_val
    normalized = []
    for x in mileage:
        normalized.append((x - min_val) / (diff))
    return np.array(normalized), diff, min_val

def estimate_price(mileage, theta0, theta1):
    price = mileage * theta1 + theta0
    return price

def calculate_theta(mileage, price, m, learning_rate, iterations):
    theta0 = 0
    theta1 = 0
    for _ in range(iterations):
        sum_error_0 = 0
        sum_error_1 = 0
        for i in range(m):
            error = estimate_price(mileage[i], theta0, theta1) - price[i]
            sum_error_0 += error
            sum_error_1 += error * mileage[i]

        tmp_theta0 = learning_rate * (1/m) * sum_error_0
        tmp_theta1 = learning_rate * (1/m) * sum_error_1

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    return theta0, theta1

def denormalize_theta(norm_theta0, norm_theta1, diff, min_val):
    theta1 = norm_theta1 / diff
    theta0 = norm_theta0 - theta1 * min_val
    return theta0, theta1

data = pd.read_csv('data.csv')

mileage = data['km']
price = data['price']
m = data.shape[0]
normalized_mileage, diff, min_val = min_max_scaling(mileage)

norm_theta0, norm_theta1 = calculate_theta(normalized_mileage, price, m, 0.1, 1000)
theta0, theta1 = denormalize_theta(norm_theta0, norm_theta1, diff, min_val)

if not os.path.exists('model'):
    os.makedirs("model")

pd.DataFrame({'theta0': [theta0], 'theta1': [theta1]}).to_csv('model/default.csv', index=False)