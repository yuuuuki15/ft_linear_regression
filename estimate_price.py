import pandas as pd

def estimate_price(mileage, theta0, theta1):
    price = mileage * theta1 + theta0
    return price

def load_model():
    try:
        model = pd.read_csv('model/default.csv')
        return model['theta0'][0], model['theta1'][0]
    except FileNotFoundError:
        return 0, 0

theta0, theta1 = load_model()

while (True):
    try:
        mileage = float(input('Enter the mileage of the car: '))
        estimated_price = estimate_price(mileage, theta0, theta1)
        print(f'Estimated price of the car: {estimated_price}')
    except ValueError:
        print('Invalid input. Please enter a numeric value for mileage.')
    except (KeyboardInterrupt, EOFError):
        print('\nExiting the program.')
        break