import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def min_max_scaling(mileage):
    min_val = mileage.min()
    max_val = mileage.max()
    diff = max_val - min_val
    normalized = []
    for x in mileage:
        normalized.append((x - min_val) / diff)
    return np.array(normalized), diff, min_val

def estimate_price(mileage, theta0, theta1):
    return mileage * theta1 + theta0

def calculate_theta(mileage, price, m, learning_rate, iterations):
    theta0 = 0
    theta1 = 0
    history = {'theta0': [theta0], 'theta1': [theta1]}
    # pure descent
    for _ in range(iterations):
        sum_error_0 = 0
        sum_error_1 = 0
        for i in range(m):
            error = estimate_price(mileage[i], theta0, theta1) - price[i]
            sum_error_0 += error
            sum_error_1 += error * mileage[i]
        tmp_theta0 = learning_rate * (1 / m) * sum_error_0
        tmp_theta1 = learning_rate * (1 / m) * sum_error_1
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        history['theta0'].append(theta0)
        history['theta1'].append(theta1)
    return theta0, theta1, history

    # coordinate descent
    # for _ in range(iterations):
    #     sum_error_0 = 0
    #     sum_error_1 = 0
    #     for i in range(m):
    #         error = estimate_price(mileage[i], theta0, theta1) - price[i]
    #         sum_error_0 += error
    #     theta0 = theta0 - learning_rate * (1 / m) * sum_error_0
    #     for i in range(m):
    #         error = estimate_price(mileage[i], theta0, theta1) - price[i]
    #         sum_error_1 += error * mileage[i]
    #     theta1 = theta1 - learning_rate * (1 / m) * sum_error_1
    #     history['theta0'].append(theta0)
    #     history['theta1'].append(theta1)
    # return theta0, theta1, history

def denormalize_theta(norm_theta0, norm_theta1, diff, min_val):
    theta1 = norm_theta1 / diff
    theta0 = norm_theta0 - theta1 * min_val
    return theta0, theta1

def visualize(history, mileage, price, diff, min_val, iterations):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Gradient descent training visualization', fontsize=14)

    axes[0, 0].plot(history['theta0'], color='steelblue')
    axes[0, 0].set_title('theta0 evolution (normalized space)')
    axes[0, 0].set_xlabel('iteration')
    axes[0, 0].set_ylabel('theta0')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['theta1'], color='seagreen')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('theta1 evolution (normalized space)')
    axes[0, 1].set_xlabel('iteration')
    axes[0, 1].set_ylabel('theta1')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['theta0'], history['theta1'],
                    color='purple', alpha=0.7)
    axes[1, 0].scatter([history['theta0'][0]], [history['theta1'][0]],
                       color='red', s=120, label='Start (0, 0)', zorder=5)
    axes[1, 0].scatter([history['theta0'][-1]], [history['theta1'][-1]],
                       color='black', s=180, marker='*', label='End', zorder=5)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('(theta0, theta1) trajectory in parameter space')
    axes[1, 0].set_xlabel('theta0')
    axes[1, 0].set_ylabel('theta1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(mileage, price, alpha=0.6, color='steelblue', label='data')
    snapshot_iters = sorted(set([
        0,
        iterations // 50,
        iterations // 10,
        iterations // 4,
        iterations // 2,
        iterations,
    ]))
    colors = plt.cm.plasma(np.linspace(0.0, 0.9, len(snapshot_iters)))
    line_x = np.array([mileage.min(), mileage.max()])
    for snap_i, color in zip(snapshot_iters, colors):
        t0_norm = history['theta0'][snap_i]
        t1_norm = history['theta1'][snap_i]
        t0_real, t1_real = denormalize_theta(t0_norm, t1_norm, diff, min_val)
        line_y = t0_real + t1_real * line_x
        axes[1, 1].plot(line_x, line_y, color=color,
                        label=f'iter {snap_i}', alpha=0.85)
    axes[1, 1].set_title('Regression line evolution (raw space)')
    axes[1, 1].set_xlabel('mileage')
    axes[1, 1].set_ylabel('price')
    axes[1, 1].legend(fontsize=8, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if not os.path.exists('images'):
        os.makedirs('images')
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'images/training_visualization_{timestamp}.png')

    plt.show()


data = pd.read_csv('data.csv')
mileage = data['km']
price = data['price']
m = data.shape[0]
normalized_mileage, diff, min_val = min_max_scaling(mileage)

iterations = 1000
norm_theta0, norm_theta1, history = calculate_theta(
    normalized_mileage, price, m, 0.1, iterations
)
theta0, theta1 = denormalize_theta(norm_theta0, norm_theta1, diff, min_val)

print(f'Final theta0: {theta0:.4f}, theta1: {theta1:.6f}')

if not os.path.exists('model'):
    os.makedirs('model')
pd.DataFrame({'theta0': [theta0], 'theta1': [theta1]}).to_csv(
    'model/default.csv', index=False
)

visualize(history, mileage, price, diff, min_val, iterations)
