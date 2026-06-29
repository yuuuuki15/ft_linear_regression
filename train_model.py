import pandas as pd
import numpy as np
import os
from pandas.errors import ParserError, EmptyDataError

# === Hyperparameters ===
LEARNING_RATE = 0.1
ITERATIONS = 1000
DATA_PATH = 'data.csv'
# =======================

def min_max_scaling(mileage):
    min_val = mileage.min()
    max_val = mileage.max()
    diff = max_val - min_val
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

if(os.path.exists(DATA_PATH)):
    try:
        data = pd.read_csv(DATA_PATH)
    except(ParserError, EmptyDataError) as e:
        print(f"Failed to read file: {e}")
        exit()
else:
    print(f"{DATA_PATH} doesn't exist.")
    exit()

try:
    mileage = pd.to_numeric(data['km'], errors="raise", downcast="float")
    price = pd.to_numeric(data['price'], errors="raise", downcast="float")
except:
    print("Failed to parse data.")
    exit()

m = data.shape[0]
normalized_mileage, diff, min_val = min_max_scaling(mileage)

norm_theta0, norm_theta1 = calculate_theta(normalized_mileage, price, m, LEARNING_RATE, ITERATIONS)
theta0, theta1 = denormalize_theta(norm_theta0, norm_theta1, diff, min_val)

if not os.path.exists('model'):
    os.makedirs("model")

pd.DataFrame({'theta0': [theta0], 'theta1': [theta1]}).to_csv('model/default.csv', index=False)