import os
import pandas as pd
import matplotlib.pyplot as plt

def load_model():
    try:
        model = pd.read_csv('model/default.csv')
        return model['theta0'][0], model['theta1'][0]
    except FileNotFoundError:
        return 0, 0

data = pd.read_csv('data.csv')
theta0, theta1 = load_model()

# print(data.head())

x = data['km']
y = data['price']
line_x = [min(data['km']), max(data['km'])]
line_y = [theta0 + theta1 * min(data['km']), theta0 + theta1 * max(data['km'])]

image_folder_name = 'images'
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
file_name = f'{image_folder_name}/price_vs_km_{timestamp}.png'

plt.scatter(x, y)
plt.plot(line_x, line_y, color='red')
plt.xlabel('Kilometers')
plt.ylabel('Price')
plt.title('Price vs Kilometers')
if (os.path.exists(image_folder_name) == False):
    os.makedirs(image_folder_name)
if (os.path.exists(file_name) == False):
    plt.savefig(f'{image_folder_name}/price_vs_km_{timestamp}.png')
else:
    print(f'File {file_name} already exists. Skipping save.')