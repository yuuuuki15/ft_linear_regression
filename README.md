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
This program will receives mileage and give you back the estimated price for that mileage.

Since theta0 and theta1 is set to 0 by default, it will return 0 whatever the mileage is.

The formula I used is below:

```
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

### 2nd program: train the model
This program will train the model, in other word, calculate theta0 and theta1 by the dateset file it received.

The formulas I used are below:

```
tmp_theta0 = learningRate * (sigma) / m
# m is a number of dataset
```