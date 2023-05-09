import numpy as np


def rmse_tracking(x_all, ref_all):
    """
    :param x_all:   Real state trajectory of dim (N, 6)
    :param ref_all: Desired reference state trajectory of dim (N, 6)
    :return:        Mean squared error
    """
    N = x_all.shape[0]

    x_pos_err = x_all[:, 0] - ref_all[:, 0]
    z_pos_err = x_all[:, 2] - ref_all[:, 2]
    err = np.sqrt(1 / N * np.sum(x_pos_err ** 2 + z_pos_err ** 2))

    return err
