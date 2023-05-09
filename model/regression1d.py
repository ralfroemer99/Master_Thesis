import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

# Number of basis functions to use
N = 11
n_points = 50

# Generate dataset of N samples
# x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
x = np.linspace(0, 4, n_points)
y = np.sin(x ** 2) + np.sqrt(x)
# y = np.array([100, 90, 80, 60, 70, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100, 96]).astype(float) / 100

reg_model = np.poly1d(np.polyfit(x, y, N - 1))

# Get impact of all the basis functions
eps_all = np.zeros(N)
for i in range(N):
    eps_all[i] = np.abs(reg_model.c[-i - 1]) * np.max(np.abs(x)) ** i

x_eval = np.linspace(0, x[-1], n_points)

fig, ax = plt.subplots(1, 3, figsize=(35, 10))
ax[0].scatter(x, y)
ax[0].plot(x_eval, reg_model(x_eval))
ax[0].set_title('Fitted polynomial')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].semilogy(np.arange(N), np.abs(np.flip(reg_model.c)))
ax[1].set_title('Coefficients')
ax[1].set_xlabel('Order')

ax[2].semilogy(np.arange(N), eps_all)
ax[2].set_title('Maximum impact')
ax[2].set_xlabel('Order')

plt.show()
# Perform linear regression using only a subset of the basis functions
