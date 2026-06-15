import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

print(data.head())

x = data['km']
y = data['price']

image_folder_name = 'images'
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
file_name = f'{image_folder_name}/price_vs_km_{timestamp}.png'

plt.scatter(x, y)
plt.xlabel('Kilometers')
plt.ylabel('Price')
plt.title('Price vs Kilometers')
if (os.path.exists(image_folder_name) == False):
    os.makedirs(image_folder_name)
if (os.path.exists(file_name) == False):
    plt.savefig(f'{image_folder_name}/price_vs_km_{timestamp}.png')
else:
    print(f'File {file_name} already exists. Skipping save.')