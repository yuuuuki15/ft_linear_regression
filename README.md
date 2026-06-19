## How to start

In this project, we will work in a virtual environment with venv.

To learn more about venv, watch [this](https://docs.python.org/3/library/venv.html)

```bash
# make venv
python3 -m venv venv

# activate environment
source venv/bin/activate

# install requirement
python3 -m pip install -r requirements.txt

# to desactivate
desactivate
```

## Mandatory part

### 1st program: predict the price
filename: estimate_price.py

This program will receives mileage and give you back the estimated price for that mileage.

Since theta0 and theta1 is set to 0 by default, it will return 0 whatever the mileage is.

The formula I used is below:

```
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

### 2nd program: train the model
filename: train_model.py

This program will train the model, in other word, calculate theta0 and theta1 by the dateset file it received.

The formulas I used are below:

```
    tmpθ₀ = learningRate × (1/m) × Σᵢ (estimatePrice(mileageᵢ) − priceᵢ)
    tmpθ₁ = learningRate × (1/m) × Σᵢ (estimatePrice(mileageᵢ) − priceᵢ) × mileageᵢ

where m is the number of data points and i ranges from 0 to m-1.

```

=== how to simultaneously update theta0 and theta1? ===

-By calculating sum of error(estimated - actual) with actual theta values.

=== Why is it necessary to simultaneously update theta0 and theta1? ===

-Actually it is not necessary to update simultaneously, it is the matter of philosophy and efficiency.
When thetas are independent, coordinate descent can be more efficient.(faster to be convergent)
When thetas are related one to another, pure descent can be more efficient.

But simultaneous update is more simple and robust in many cases.


## Bonus part

### plotting the data into a graph / plotting the line(using theta0 and theta1)
filename: visualize.py

This program will show you the graph of given data(./data.csv) and

### calculate the precision
filename: precision.py

This program will calculate the precision of algorithm.

The formulas I used are below:

```python
# For MAE(Mean absolute error)
return np.sum(np.abs(predicted - actual)) / m

# For RMSE(Root mean square deviation)
return np.sqrt(np.sum((predicted - actual) ** 2) / m)

# For R²(R squared)
avr = np.mean(actual)
left = np.sum((predicted - actual) ** 2)
right = np.sum((avr - actual) ** 2)
return 1 - (left / right)
```
#### Difference between these algorithm

MAE is easy and simple way. Lower is better.

RMSE is more sensitive to outliers. Lower is better.

R2 is suitable to understand the overall fit of the model to simple average baseline. Higher is better.

resources:
https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e

