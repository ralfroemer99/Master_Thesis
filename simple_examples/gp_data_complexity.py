import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
# matplotlib inline
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100


# Calculate error bound
def error_bound():
    tau = 0.01
    Lf = 4
    dx = 1
    r0 = 2
    delta = 0.01
    beta = 2 * dx * np.log(1 + r0 / tau) - 2 * np.log(delta)
    return beta


def sigma_bound(N, sig_f, l, sig_n, L):
    return np.sqrt(sig_f ** 2 - sig_f ** 4 / (sig_f ** 2 + sig_n ** 2 / N) * np.exp(- L ** 2 / (4 * l ** 2 * (N - 1)
                                                                                                ** 2)))


print(error_bound())


# Generate non-linear function.
def f(x):
    f = np.sin(np.pi * x) + np.cos(np.pi * x) + np.sin(2 * np.pi * x)
    # f = np.ones_like(x)
    return f


# Set dimension.
d = 1
# Number of training points.
n = np.arange(start=10, stop=210, step=10)
# Upper bound of the training set.
L = 2

# Maximum uncertainty
unc_max = np.zeros(len(n))
sigma_max = np.zeros(len(n))
kernel_params = np.zeros((len(n), 2))

for i in range(len(n)):
    # Generate training features.
    x = np.linspace(start=0, stop=L, num=n[i])
    X = x.reshape(n[i], d)
    # Error standard deviation.
    sigma_n = 0.1
    # Errors.
    epsilon = np.random.normal(loc=0, scale=sigma_n, size=n[i])

    f_x = f(x)

    # Observed target variable.
    y = f_x + epsilon

    # Define kernel parameters.
    l = 0.1
    sigma_f = 2

    # Test set
    n_star = 1000
    x_star = np.linspace(start=0, stop=L, num=n_star)
    X_star = x_star.reshape(n_star, d)

    # Define kernel object.
    # kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) \
    #          * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))
    kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds="fixed") \
             * RBF(length_scale=l, length_scale_bounds="fixed")
    # Define GaussianProcessRegressor object.
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=10, )

    # FIT to data using Maximum Likelihood Estimation of the parameters.
    gp.fit(X, y)
    # Make the prediction on test set.
    y_pred = gp.predict(X_star)
    # Generate samples from posterior distribution.
    y_hat_samples = gp.sample_y(X_star, n_samples=n_star)
    # Compute the mean of the sample.
    y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
    # Compute the standard deviation of the sample.
    y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()

    unc_max[i] = np.max(y_hat_sd)

    sigma_max[i] = sigma_bound(n[i], sigma_f, l, sigma_n, L)
    kernel_params[i, 0] = np.sqrt(gp.kernel_.k1.constant_value)
    kernel_params[i, 1] = gp.kernel_.k2.length_scale

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(n, unc_max, color='g')
ax.plot(n, sigma_max, color='r')
ax.set_xlabel('Number of training samples')
ax.set_ylabel('Maximum uncertainty')
plt.show()

# PLOT
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
# Plot training data.
ax[0].scatter(x, y, label='training data')
# Plot "true" linear fit.
ax[0].plot(x_star, f(x_star), color='red', label='f(x)', )
ax[0].plot(x_star, y_pred, color='green', label='pred')
# Plot corridor.
ax[0].fill_between(
    x=x_star,
    y1=(y_hat - 2 * y_hat_sd),
    y2=(y_hat + 2 * y_hat_sd),
    color='green',
    alpha=0.3,
    label='Credible Interval'
)
ax[1].plot(x_star, y_hat_sd, label='Standard deviation')
# Plot prediction.

ax[0].legend(loc='lower left')
plt.show()
