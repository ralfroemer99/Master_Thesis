import numpy as np
import matplotlib as mp
import matplotlib.backends.backend_agg as mpbe
import matplotlib.animation as animation

h_prop = 0.01
d_prop = 0.06


def plot_robot(ax, x_all, u_all, l, t_all):
    # Decide how many samples to jump before
    dt = t_all[1] - t_all[0]
    n_skip = int(0.2 / dt)

    for i in range(len(t_all)):
        if i % n_skip == 0:
            list_of_lines = []
            # Create the robot
            # the main frame
            line, = ax.plot([], [], 'k', lw=6)
            list_of_lines.append(line)
            # the left propeller
            line, = ax.plot([], [], 'b', lw=4)
            list_of_lines.append(line)
            # the right propeller
            line, = ax.plot([], [], 'b', lw=4)
            list_of_lines.append(line)
            # the left thrust
            line, = ax.plot([], [], 'r', lw=2)
            list_of_lines.append(line)
            # the right thrust
            line, = ax.plot([], [], 'r', lw=2)
            list_of_lines.append(line)

            for _ in list_of_lines:  # reset all lines
                _.set_data([], [])

            x = x_all[i, 0]
            z = x_all[i, 2]
            theta = x_all[i, 4]
            trans = np.array([[x, x], [z, z]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            main_frame = np.array([[-l / 2, l / 2], [0, 0]])
            main_frame = rot @ main_frame + trans

            left_propeller = np.array([[-d_prop / 2, d_prop / 2],
                                       [0, 0]])
            left_propeller = rot @ left_propeller + trans + np.array([[- l / 2 * np.cos(theta) - h_prop * np.sin(theta),
                                                                       - l / 2 * np.cos(theta) - h_prop * np.sin(
                                                                           theta)],
                                                                      [- l / 2 * np.sin(theta) + h_prop * np.cos(theta),
                                                                       - l / 2 * np.sin(theta) + h_prop * np.cos(
                                                                           theta)]])

            right_propeller = np.array([[-d_prop / 2, d_prop / 2],
                                        [0, 0]])
            right_propeller = rot @ right_propeller + trans + np.array([[l / 2 * np.cos(theta) - h_prop * np.sin(theta),
                                                                         l / 2 * np.cos(theta) - h_prop * np.sin(
                                                                             theta)],
                                                                        [l / 2 * np.sin(theta) + h_prop * np.cos(theta),
                                                                         l / 2 * np.sin(theta) + h_prop * np.cos(
                                                                             theta)]])

            left_thrust = np.array([[-l / 2, -l / 2], [h_prop, h_prop + u_all[i, 0] * 0.04]])
            left_thrust = rot @ left_thrust + trans

            right_thrust = np.array([[l / 2, l / 2], [h_prop, h_prop + u_all[i, 1] * 0.04]])
            right_thrust = rot @ right_thrust + trans

            list_of_lines[0].set_data(main_frame[0, :], main_frame[1, :])
            list_of_lines[1].set_data(left_propeller[0, :], left_propeller[1, :])
            list_of_lines[2].set_data(right_propeller[0, :], right_propeller[1, :])
            list_of_lines[3].set_data(left_thrust[0, :], left_thrust[1, :])
            list_of_lines[4].set_data(right_thrust[0, :], right_thrust[1, :])


def animate_robot(x_all, u_all, l, dt=0.01):
    """
    This function makes an animation showing the behavior of the quadrotor
    takes as input the result of a simulation (with dt=0.01s)
    """
    min_dt = 0.1
    if (dt < min_dt):
        steps = int(min_dt / dt)
        use_dt = int(np.round(min_dt * 1000))
    else:
        steps = 1
        use_dt = int(np.round(dt * 1000))

    # what we need to plot
    plotx = x_all[::steps, :]
    plotx = plotx[:-1, :]
    plotu = u_all[::steps, :]

    fig = mp.figure.Figure(figsize=[8.5, 8.5])
    mpbe.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=[-4, 4], ylim=[-4, 4])
    ax.grid()

    list_of_lines = []

    # Create the robot
    # the main frame
    line, = ax.plot([], [], 'k', lw=6)
    list_of_lines.append(line)
    # the left propeller
    line, = ax.plot([], [], 'b', lw=4)
    list_of_lines.append(line)
    # the right propeller
    line, = ax.plot([], [], 'b', lw=4)
    list_of_lines.append(line)
    # the left thrust
    line, = ax.plot([], [], 'r', lw=1)
    list_of_lines.append(line)
    # the right thrust
    line, = ax.plot([], [], 'r', lw=1)
    list_of_lines.append(line)

    def _animate(i):
        for _ in list_of_lines:  # reset all lines
            _.set_data([], [])

        x = plotx[i, 0]
        y = plotx[i, 2]
        theta = plotx[i, 4]
        trans = np.array([[x, x], [y, y]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        main_frame = np.array([[-l, l], [0, 0]])
        main_frame = rot @ main_frame + trans

        left_propeller = np.array([[-1.3 * l, -0.7 * l], [0.1, 0.1]])
        left_propeller = rot @ left_propeller + trans

        right_propeller = np.array([[1.3 * l, 0.7 * l], [0.1, 0.1]])
        right_propeller = rot @ right_propeller + trans

        left_thrust = np.array([[l, l], [0.1, 0.1 + plotu[i, 0] * 0.04]])
        left_thrust = rot @ left_thrust + trans

        right_thrust = np.array([[-l, -l], [0.1, 0.1 + plotu[i, 1] * 0.04]])
        right_thrust = rot @ right_thrust + trans

        list_of_lines[0].set_data(main_frame[0, :], main_frame[1, :])
        list_of_lines[1].set_data(left_propeller[0, :], left_propeller[1, :])
        list_of_lines[2].set_data(right_propeller[0, :], right_propeller[1, :])
        list_of_lines[3].set_data(left_thrust[0, :], left_thrust[1, :])
        list_of_lines[4].set_data(right_thrust[0, :], right_thrust[1, :])

        return list_of_lines

    def _init():
        return _animate(0)

    ani = animation.FuncAnimation(fig, _animate, np.arange(0, len(plotx[0, :])), interval=use_dt, blit=True,
                                  init_func=_init)
    # plt.close(fig)
    # plt.close(ani._fig)
    IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
