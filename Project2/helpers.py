from math import pi
from math import sqrt
from torch import tensor
from torch import empty
from torch import empty
from torch import set_grad_enabled
from torch import Tensor
import math
# remove before submission
from torch import float64
import torch
set_grad_enabled(False)

def inside_circle(point):
        CENTER_X, CENTER_Y = 0.5, 0.5
        RADIUS_SQUARED = 1/(2*pi)
        res = [0, 0]
        if ((point[0] - CENTER_X).pow(2) + (point[1] - CENTER_Y).pow(2)) <= RADIUS_SQUARED:
            res[0] = 1
        else:
            res[1] = 1
        return res

def generate_circle_data(N_POINTS):
        points = empty(N_POINTS, 2).uniform_()
        labels = []
        for point in points:
            labels.append(inside_circle(point))

        labels = tensor(labels)
        return points, labels