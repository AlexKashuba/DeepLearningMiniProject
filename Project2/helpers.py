from math import pi
from torch import empty
from torch import set_grad_enabled

set_grad_enabled(False)

CENTER_X, CENTER_Y = 0.5, 0.5
RADIUS_SQUARED = 1/(2*pi)


def nb_errors(output, label):
    """Calculate the number of errors across a minibatch"""
    _, predicted = output[0].max(1)
    _, real = label.max(1)
    return (predicted - real).abs().sum()


def inside_circle(point, res_tensor):
    if ((point[0] - CENTER_X).pow(2) + (point[1] - CENTER_Y).pow(2)) <= RADIUS_SQUARED:
        res_tensor[0] = 1
    else:
        res_tensor[1] = 1


def generate_circle_data(N_POINTS):
    points = empty(N_POINTS, 2).uniform_(0, 1)
    labels = empty(N_POINTS, 2).zero_()

    for i in range(N_POINTS):
        inside_circle(points[i], labels[i])

    return points, labels
