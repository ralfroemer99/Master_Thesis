import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from numpy.random import Generator
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.integrate import odeint, ode

n_trials = 10

# Define system and simulation
T_sim = 20
dt = 0.1
xlim_low = [-1.5, -1.5]
xlim_high = [1.5, 1.5]
gains = [-2, -2]
Q = np.eye(2)
x0 = np.array([0, 1])

# GP specifications
N0 = np.arange(start=10, stop=210, step=10)
kc = [0, 0.001, 0.002]
computation_time = kc * N0[0]
noise = 0.01
l = 0.2
sigma_f = 1


def f(x):
    return np.sin(x[0]) + 0.5 / (1 + np.exp(x[1] / 10))


def controller(t, x, xd, dxd, fhat, t_k, Ts):
    e = np.zeros(2)
    e[0], e[1] = x[0] - xd[0], x[1] - xd[1]
    delay = 0
    if Ts != 0:
        if t <= Ts:
            f_comp = 0
        elif Ts < t <= 2 * Ts or t <= 2 * dt:
            delay = t
            x_old = x0
            f_comp = fhat.predict(x_old.reshape((1, 2)))
        else:
            x_old = np.zeros(2)
            delay = Ts + np.mod(t, Ts)
            x_old[0] = np.interp(t - delay, np.array(t_sol), np.array(x_sol)[:, 0])
            x_old[1] = np.interp(t - delay, np.array(t_sol), np.array(x_sol)[:, 1])
            f_comp = fhat.predict(x_old.reshape((1, 2)))
    else:
        f_comp = fhat.predict(x.reshape((1, 2)))

    u = dxd[1] - f_comp + gains[0] * e[0] + gains[1] * e[1]
    return u, f_comp, delay


# Define dynamics
def dxdt(t, x, p):
    fhat, Ts, t_k = p
    xd = np.array([np.sin(t), np.cos(t)])
    dxd = np.array([np.cos(t), -np.sin(t)])
    dx = np.zeros(2)
    # Detect sampling instant
    if t - t_k >= Ts:
        t_k = t_k + Ts

    u, _, _ = controller(t, x, xd, dxd, fhat, t_k, Ts)
    dx[0] = x[1]
    dx[1] = f(x) + u
    return dx


error_max = np.zeros((len(kc), n_trials, len(N0)))
error_rmse = np.zeros((len(kc), n_trials, len(N0)))
for h in range(len(kc)):
    for k in range(len(N0)):
        for i in range(n_trials):
            # Create training samples
            X_train = np.random.default_rng().uniform(low=xlim_low, high=xlim_high, size=(N0[k], 2))
            Y_train = np.zeros(N0[k])
            for _ in range(N0[k]):
                Y_train[_] = f(X_train[_]) + np.random.default_rng().normal(scale=noise)

            # Plot training data
            # if len(N0) == 1:
            #     fig, ax = plt.subplots(figsize=(10, 10))
            #     ax.scatter(X_train[:, 0], X_train[:, 1])
            #     plt.show()

            # Train GP model
            # kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=[1e-2, 1e2]) * RBF(
            #     length_scale=l, length_scale_bounds=[1e-2, 1e2])
            kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds="fixed") * RBF(
                length_scale=l, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2, n_restarts_optimizer=10, )
            gp.fit(X_train, Y_train)

            # Simulate system
            computation_time = kc[h] * N0[k]
            sampling_instant = 0
            params = gp, computation_time, sampling_instant
            r = ode(dxdt)
            r.set_initial_value(x0, 0).set_f_params(params)
            t_sol, x_sol = [], []
            while r.successful() and r.t < T_sim:
                r.integrate(r.t + dt)
                x_sol.append(r.y)
                t_sol.append(r.t)
            t_sol = np.array(t_sol)
            x_sol = np.array(x_sol)

            # Get maximum error
            xd = np.vstack((np.sin(t_sol), np.cos(t_sol))).T
            error_max[h, i, k] = np.max(np.linalg.norm(xd - x_sol, axis=1))
            error_rmse[h, i, k] = np.sqrt(np.sum(np.linalg.norm(xd - x_sol, axis=1) ** 2) / x_sol.shape[0])

# Get and plot input trajectory
u_sol = np.zeros_like(t_sol)
fhat_sol = np.zeros_like(t_sol)
delay_sol = np.zeros_like(t_sol)
t_k = 0
for _ in range(len(u_sol)):
    xd = np.array([np.sin(t_sol[_]), np.cos(t_sol[_])])
    dxd = np.array([np.cos(t_sol[_]), -np.sin(t_sol[_])])
    # Detect sampling instant
    if t_sol[_] - t_k >= computation_time:
        t_k = t_k + computation_time
    u_sol[_], fhat_sol[_], delay_sol[_] = controller(t_sol[_], x_sol[_], xd, dxd, gp, t_k, computation_time)

# fig, ax = plt.subplots(figsize=(15, 8))
# ax.plot(t_sol, delay_sol)
# plt.show()

# Plot error
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(N0, np.mean(error_rmse[0], axis=0), 'r', label='$k_\\mathrm{c} = 0')
ax.plot(N0, np.mean(error_rmse[1], axis=0), 'g', label='$k_\\mathrm{c} = 0.001')
ax.plot(N0, np.mean(error_rmse[2], axis=0), 'b', label='$k_\\mathrm{c} = 0.002')
ax.set_xlabel('$N_0$')
ax.set_ylabel('Mean tracking error')
# ax.legend()
tikzplotlib.save('../tikz_plots/gp_feedback_lin_tracking_error.tex')
plt.show()

# Plot trajectory
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].plot(x_sol[:, 0], x_sol[:, 1])
ax[0].set_xlabel('$x_1$')
ax[0].set_xlabel('$x_2$')
ax[1].plot(t_sol, x_sol[:, 0], 'r', label='$x_1$')
ax[1].plot(t_sol, x_sol[:, 1], 'b', label='$x_2$')
ax[1].plot(t_sol, np.sin(t_sol), 'r--', label='$x_1$')
ax[1].plot(t_sol, np.cos(t_sol), 'b--', label='$x_2$')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$x$')
plt.show()
