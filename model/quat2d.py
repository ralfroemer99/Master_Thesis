import numpy as np
import scipy.signal
from scipy.integrate import odeint

g = 9.81


class Quadrotor:
    def __init__(self, mass=0.2, length=0.2, inertia=0.1):
        self.MASS = mass
        self.LENGTH = length
        self.INERTIA = inertia
        self.x_lim = np.array([[-5, 5],
                               [-5, 5],
                               [-5, 5],
                               [-5, 5],
                               [-np.pi, np.pi],
                               [-1, 1]])
        self.u_lim = np.array([[-5, 5],
                               [-5, 5]])

    def dynamics_nonlinear_cont(self, x, t, u):
        """
        This function is an implementation of the simplified 2D quadrotor dynamics
        :param x:   State of dim 6
        :param u:   Control input of dim 2
        :return:    Time derivative of the state of dim 6
        """
        dx_dt = np.zeros(6)
        # x-position and velocity
        dx_dt[0] = x[1]
        dx_dt[1] = -1 / self.MASS * np.sin(x[4]) * (u[0] + u[1])
        # z-position and velocity
        dx_dt[2] = x[3]
        dx_dt[3] = 1 / self.MASS * np.cos(x[4]) * (u[0] + u[1]) - g
        # Orientation and angular velocity
        dx_dt[4] = x[5]
        dx_dt[5] = 1 / self.INERTIA * self.LENGTH / 2 * (u[0] - u[1])

        return dx_dt

    def dynamics_nonlinear_disc(self, x, t, u, dt=0.1):
        """
        This function is discretized version of the simplified 2D quadrotor dynamics
        :param x:   Current state of dim 6
        :param u:   Control input of dim 2, kept constant during sampling interval
        :param dt:  Sampling time
        :return:    Time derivative of the state of dim 6
        """
        t = np.linspace(0, dt, 101)
        sol = odeint(self.dynamics_nonlinear_cont(), x, t, args=(u,))
        x_next = sol[-1, :]
        return x_next

    def dynamics_linear_cont(self, x, u):
        """
        :param x:   State about which to linearize of dim 6
        :param u:   Input about which to linearize of dim 2
        :return:    A and B matrices of the linearized system of dims (6, 6) and (6, 2)
        """
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1 / self.MASS * np.cos(x[4]) * (u[0] + u[1]), 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, -1 / self.MASS * np.sin(x[4]) * (u[0] + u[1]), 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      ])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [1 / self.MASS, 1 / self.MASS],
                      [0, 0],
                      [self.LENGTH / self.INERTIA, -self.LENGTH / self.INERTIA],
                      ])
        return A, B

    def dynamics_linear_disc(self, x, u, dt=0.01):
        """
        :param x:   State about which to linearize of dim 6
        :param u:   Input about which to linearize of dim 2
        :param dt:  Sampling time for the discretization
        :return:    A and B matrices of the linearized discrete-time system of dims (6, 6) and (6, 2)
        :return:
        """
        Ac, Bc = self.dynamics_linear_cont(x, u)
        Ad, Bd, _, _, _ = scipy.signal.cont2discrete((Ac, Bc, [], []), dt)
        return Ad, Bd

    def random_state_input_pair(self, N):
        """
        :param N:   Number of samples to generate
        :return:    N random state-input pairs within the system constraints
        :return:
        """
        x_rand = np.random.uniform(self.x_lim[:, 0], self.x_lim[:, 1], size=(N, 6))
        u_rand = np.random.uniform(self.u_lim[:, 0], self.u_lim[:, 1], size=(N, 2))
        return x_rand, u_rand
