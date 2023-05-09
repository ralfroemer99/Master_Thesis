import numpy as np
import matplotlib as mp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from animation.animation import animate_robot, plot_robot
from reference_trajectories import define_reference
from eval.evaluation_metrics import rmse_tracking
from model.quat2d import Quadrotor
from controllers.controllers import lqr_disc, linear_mpc

font = {'size': 20}

mp.rc('font', **font)

# Simulation parameters
g = 9.81
SIM_TIME = 10
DELTA_T = 0.1


def sim_forward(sys, x, u, dt):
    """
    :param sys: System object specifying the nonlinear dynamics
    :param x:   Initial state of dim n
    :param u:   Control input (constant during the sampling interval) of dim m
    :param dt:  Sampling time
    :return:    Final state after a time of dt
    """
    t = np.linspace(0, dt, 101)
    sol = odeint(sys.dynamics_nonlinear, x, t, args=(u,))
    return sol[-1, :]


def main():
    # Define simulation parameters
    sim_timesteps = int(SIM_TIME / DELTA_T)
    x_init = np.array([0, 0, 1.5, 0, 0, 0])

    robot = Quadrotor()

    x_all = np.zeros((sim_timesteps, 6))
    u_all = np.zeros((sim_timesteps, 2))
    t_all = np.arange(0, sim_timesteps * DELTA_T, DELTA_T)

    # Reference trajectory
    ref_all = define_reference(which='diagonal_line', x_init=x_init, timesteps=sim_timesteps)

    # Equilibrium and input from previous timestep
    u_ep = np.array([robot.MASS * g / 2, robot.MASS * g / 2])
    u_prev = u_ep

    # Execute simulation
    x = x_init
    for i in range(sim_timesteps):
        x_ref = ref_all[i]
        # Compute control action
        # u = lqr_disc(robot, x, ref_all[i], u_ep, DELTA_T)
        u = linear_mpc(robot, x, ref_all[i:], u_ep, DELTA_T)
        u = u + u_ep
        # Store x and u
        x_all[i, :] = x
        u_all[i, :] = u

        # Simulate system with computed control action
        x = sim_forward(robot, x, u, DELTA_T)

    # Calculate position tracking error
    err = rmse_tracking(x_all, ref_all)
    print(err)

    # Plot robot
    fig, ax = plt.subplots(1, 3, figsize=(30, 15))
    # Show trajectory in plot
    plot_robot(ax[0], x_all, u_all, robot.LENGTH, t_all)
    ax[0].plot(x_all[:, 0], x_all[:, 2], label='$(x,z)$')
    ax[0].plot(ref_all[:, 0], ref_all[:, 2], 'k--', label='$(x_\\mathrm{r},z_\\mathrm{r})$')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$z$')
    ax[0].set_xlim([-0.5, 2.5])
    ax[0].set_ylim([0, 3])
    ax[0].legend()
    ax[1].plot(t_all, x_all[:, 0], 'b', label='$x$')
    ax[1].plot(t_all, ref_all[:, 0], 'b:', label='$x_\\mathrm{r}$')
    ax[1].plot(t_all, x_all[:, 1], 'b--', label='$\\dot{x}$')
    ax[1].plot(t_all, x_all[:, 2], 'g', label='$z$')
    ax[1].plot(t_all, ref_all[:, 2], 'g:', label='$z_\\mathrm{r}$')
    ax[1].plot(t_all, x_all[:, 3], 'g--', label='$\\dot{z}$')
    ax[1].plot(t_all, x_all[:, 4], 'r', label='$\\theta$')
    ax[1].plot(t_all, x_all[:, 5], 'r--', label='$\\dot{\\theta}$')
    ax[1].set_xlabel('$t$')
    ax[1].legend()
    ax[2].plot(t_all, u_all[:, 0], 'b', label='$u_1$ (left)')
    ax[2].plot(t_all, u_all[:, 1], 'b--', label='$u_2$ (right)')
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('$u$')
    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
