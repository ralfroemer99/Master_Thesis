import numpy as np
import control
import math
import tikzplotlib
import time
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.integrate import odeint
from cvxpy import Variable, Problem, Minimize, Parameter, quad_form, OSQP

font = {'size': 20}

mp.rc('font', **font)

DT_SIM = 0.01
Q = np.diag([1, 1])
R = 0.01
SIM_TIME = 10
SIM_TIMESTEPS = int(SIM_TIME / DT_SIM)
Ac = np.array([[0, 1],
               [2, -1]])
Bc = np.array([0, 1]).reshape((2, 1))


def dynamics_cont(x, t, u):
    dx = Ac @ x + Bc @ u
    return dx


def sim_forward(x, u, Tc):
    t = np.linspace(0, Tc, int(round(Tc/DT_SIM)) + 1)
    sol = odeint(dynamics_cont, x, t, args=(u,))
    return sol


def controller_model(Ac, Bc, Tc, dist):
    if dist == 0:       # Controller knows Ac and Bc exactly
        Ac_nom, Bc_nom = Ac, Bc
    elif dist == 1:  # Parametric disturbance that is independent of Tc
        Ac_nom = Ac + np.array([[0, 0],
                                [-1, 0]])
        Bc_nom = Bc + np.array([[0], [0.5]])
    elif dist == 2:  # Parametric disturbance that depends on Tc
        Ac_nom = Ac + np.array([[0, 0],
                                [-1 + 1 * np.minimum(1, Tc), 0]])
        Bc_nom = Bc + np.array([[0], [0.5 - 0.5 * np.minimum(1, Tc)]])

    # Discretize system
    Ad, Bd, _, _, _ = cont2discrete((Ac_nom, Bc_nom, [], []), dt=Tc)

    return Ad, Bd


def linear_mpc(Ad, Bd, x0, Tc):
    # Ensure that the prediction horizon has a length of at least one second
    Nhor = math.ceil(2 / Tc)
    [nx, nu] = Bd.shape

    # State constraints
    xmin = np.array([-1, -4])
    xmax = np.array([6, 1.5])
    umin = -20
    umax = 20

    # Define problem
    u = Variable((Nhor, nu))
    x = Variable((Nhor + 1, nx))
    x_init = Parameter(nx)
    objective = 0
    constraints = [x[0] == x_init]
    for k in range(Nhor):
        # Stage cost
        objective += quad_form(x[k], Q) + quad_form(u[k], np.array([[R]]))
        # Dynamics
        constraints += [x[k + 1] == Ad @ x[k] + Bd @ u[k]]
        # State and input constraints
        constraints += [xmin <= x[k], x[k] <= xmax]
        constraints += [umin <= u[k], u[k] <= umax]
    # Terminal cost
    objective += quad_form(x[-1], Q)
    prob = Problem(Minimize(objective), constraints)
    # Initialize initial state with current state
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    return u[0].value


def eval_cost(x_traj, u_traj):
    cost = 0
    for i in range(x_traj.shape[0]):
        cost += 0.5 * x_traj[i].reshape(1, 2) @ Q @ x_traj[i].reshape(2, 1) * DT_SIM +\
                0.5 * u_traj[i] * R * u_traj[i] * DT_SIM
    return cost


def main():
    x_init = np.array([5, 1])
    # Try out different model inaccuracies
    dist = [0]
    # Define controller type (LQR or MPC)
    which_controller = 'MPC'
    # Try out different sampling times
    Tc = np.arange(0.02, 1, 0.01)
    rmse_all = np.zeros((len(Tc), len(dist)))
    cost_all = np.zeros((len(Tc), len(dist)))
    comp_time_all = np.zeros((len(Tc), len(dist)))

    for d in range(len(dist)):
        for n in range(len(Tc)):
            x_all = np.zeros((SIM_TIMESTEPS, 2))
            u_all = np.zeros(SIM_TIMESTEPS)
            t_all = np.arange(0, SIM_TIMESTEPS * DT_SIM, DT_SIM)

            # Simulation
            x = x_init

            # Controller model
            Ad, Bd = controller_model(Ac, Bc, Tc[n], dist=dist[d])

            Kd, _, _ = control.dlqr(Ad, Bd, Q, R)
            n_timesteps_controller = int(SIM_TIME / Tc[n])
            a = int(round(Tc[n] / DT_SIM))
            comp_time_tmp = 0
            for i in range(n_timesteps_controller + 1):
                if which_controller == 'LQR':
                    u = -Kd @ x
                elif which_controller == 'MPC':
                    start = time.time()
                    u = linear_mpc(Ad, Bd, x, Tc[n])
                    comp_time_tmp += (time.time() - start) / (n_timesteps_controller + 1)

                x_next = sim_forward(x, u, Tc[n])
                idx_low = min(i * a, SIM_TIMESTEPS - 1)
                idx_up = min((i + 1) * a, SIM_TIMESTEPS)
                x_all[idx_low:idx_up] = x_next[:idx_up - idx_low]
                u_all[idx_low:idx_up] = u
                x = x_next[-1]

            # Save computation time and cost of current run
            comp_time_all[n, d] = comp_time_tmp
            rmse_all[n, d] = np.sqrt(1 / SIM_TIMESTEPS * np.sum(np.linalg.norm(x_all, axis=1)**2))
            cost_all[n, d] = eval_cost(x_all, u_all)

    if len(Tc) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        ax.plot(1 / Tc, np.mean(comp_time_all, axis=1))
        ax.plot(1 / Tc, Tc)
        ax.set_xlabel('Control frequency')
        ax.set_ylabel('Computation time')
        tikzplotlib.save('../tikz_plots/2d_lin_comp_time_mpc_Thor2.tex')
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        for d in range(len(dist)):
            ax.plot(1 / Tc, cost_all[:, d], label='dist = ' + str(dist[d]))
        ax.set_xlabel('Control frequency')
        ax.set_ylabel('Cost')
        ax.set_ylim([15, 30])
        # tikzplotlib.save('../tikz_plots/2d_lin_cost_mpc_Thor1.tex')
        plt.show()

    # Plot robot
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    # Show trajectory in plot
    ax.plot(x_all[:, 0], x_all[:, 1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('(x_1, x_2)')
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    ax.plot(t_all, x_all[:, 0], 'b', label='$x_1$')
    ax.plot(t_all, x_all[:, 1], 'r', label='$x_2$')
    ax.plot(t_all, u_all, 'g', label='$u$')
    ax.set_xlabel('$t$')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()