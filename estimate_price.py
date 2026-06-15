
def estimate_price(mileage):
    theta0 = 0
    theta1 = 0
    price = mileage * theta1 + theta0
    return price

while (True):
    try:
        mileage = float(input('Enter the mileage of the car: '))
        estimated_price = estimate_price(mileage)
        print(f'Estimated price of the car: {estimated_price}')
    except ValueError:
        print('Invalid input. Please enter a numeric value for mileage.')
    except (KeyboardInterrupt, EOFError):
        print('\nExiting the program.')
        break