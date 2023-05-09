import numpy as np


def define_reference(which='circle_pos', x_init=[0, 0, 1, 0, 0, 0], timesteps=100):
    """
    :param which:       Type of the desired trajectory (circle_pos)
    :param pos_init:    Initial position
    :param timesteps:   Number of discretization steps for the reference trajectory
    :return:
    """
    if which == 'vertical_line':
        length = 1
        ref = np.zeros((timesteps, 6))
        theta_all = np.linspace(0, 2 * np.pi, timesteps)
        for i in range(timesteps):
            ref[i, 2] = x_init[2] + i / timesteps * length
    elif which == 'horizontal_line':
        length = 1
        ref = np.zeros((timesteps, 6))
        theta_all = np.linspace(0, 2 * np.pi, timesteps)
        for i in range(timesteps):
            ref[i, 0] = x_init[0] + i / timesteps * length
            ref[i, 2] = x_init[2]
    elif which == 'diagonal_line':
        length = 2
        ref = np.zeros((timesteps, 6))
        theta_all = np.linspace(0, 2 * np.pi, timesteps)
        for i in range(timesteps):
            ref[i, 0] = x_init[0] + i / timesteps * 0.2 * length
            ref[i, 2] = x_init[2] + i / timesteps * length
            # ref[i, 4] = -np.arctan(0.1)

    elif which == 'circle':
        radius = 1
        ref = np.zeros((timesteps, 6))
        theta_all = np.linspace(0, 2 * np.pi, timesteps)
        for i in range(timesteps):
            ref[i, 0] = x_init[0] + radius * (1 - np.cos(theta_all[i]))
            ref[i, 2] = x_init[2] + radius * np.sin(theta_all[i])
    if which == 'circle_hor':
        radius = 1
        ref = np.zeros((timesteps, 6))
        theta_all = np.linspace(0, 2 * np.pi, timesteps)
        for i in range(timesteps):
            ref[i, 0] = x_init[0] + np.sin(theta_all[i]) * radius
            ref[i, 2] = x_init[2] + radius * (1 - np.cos(theta_all[i]))

    return ref
