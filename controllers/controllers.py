import control
import numpy as np
from cvxpy import Variable, Problem, Minimize, Parameter, quad_form, OSQP

Q = np.diag(np.array([100, 0, 100, 0, 0, 0]))
R = np.diag(np.array([1, 1]))


def lqr_disc(sys, x, x_ref, u_ref, dt):
    """
    :param sys:     System object specifying the dynamics
    :param x:       Current state of dim Nx
    :param x_ref:   Reference state of dim Nx
    :param u_ref:   Reference input of dim Nu
    :param dt:      Sampling time for the discretization of the dynamics
    :return:        Desired control input of dim Nu
    """
    Ad, Bd = sys.dynamics_linear_disc(x_ref, u_ref, dt)
    K, _, _ = control.dlqr(Ad, Bd, Q, R)
    return -K @ (x - x_ref) + u_ref


def linear_mpc(sys, x0, x_ref, u_ref, dt):
    """
    :param sys:     System object specifying the dynamics
    :param x0:      Current state of dim Nx
    :param x_ref:   Reference trajectory of dim (Nhor, Nx)
    :param u_ref:   Reference input of dim Nu
    :param dt:      Sampling time
    :return:        First value of the calculated input trajectory of dim Nu
    """
    Nhor = 10
    Ad, Bd = sys.dynamics_linear_disc(x_ref[0], u_ref, dt)
    [nx, nu] = Bd.shape

    # Check if reference trajectory of length N is applied
    if x_ref.shape[0] < Nhor:
        x_ref_last = x_ref[-1].reshape((1, nx))
        x_ref = np.vstack([x_ref, np.repeat(x_ref_last, Nhor-x_ref.shape[0], axis=0)])

    # Define problem
    u = Variable((Nhor, nu))
    x = Variable((Nhor + 1, nx))
    x_init = Parameter(nx)
    objective = 0
    constraints = [x[0] == x_init]
    for k in range(Nhor):
        # Stage cost
        objective += quad_form(x[k] - x_ref[k], Q) + quad_form(u[k], R)
        # Dynamics
        constraints += [x[k + 1] == Ad @ x[k] + Bd @ u[k]]
        # State and input constraints
        # constraints += [xmin <= x[:, k], x[:, k] <= xmax]
        # constraints += [umin <= u[:, k], u[:, k] <= umax]
    # Terminal cost
    objective += quad_form(x[-1] - x_ref[k], Q)
    prob = Problem(Minimize(objective), constraints)
    # Initialize initial state with current state
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    return u[0].value
