from torch import empty
from torch import set_grad_enabled

# remove before submission
from torch import float64
import torch
from math import sqrt
set_grad_enabled(False)
from torch.nn.init import xavier_normal_, xavier_normal



class Module(object):

    def forward(self, *_input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    def __init__(self, in_features, hidden_units, name=None):
        self.name = name
        self._input = None
        self.in_features = in_features
        self.hidden_units = hidden_units
        xavier_init = sqrt(6/(in_features + hidden_units))
        self.weights = empty(hidden_units, in_features).uniform_(-xavier_init, xavier_init)
        self.grad_w = empty(self.weights.size()).zero_()
        self.bias = torch.empty(self.hidden_units).normal_(0, 1e-3)
        self.grad_b = empty(self.bias.size()).zero_()

    def forward(self, *_input):
        if self.name:
            print("Layer: {}".format(self.name))
        self._input = _input[0]
        res = (self._input.mm(self.weights.T) + self.bias, )
        return res

    def backward(self, *gradwrtoutput):
        self.grad_b.add_(gradwrtoutput[0].sum(0))
        d_in = self.weights.t().mm(gradwrtoutput[0].t())
        d_w = gradwrtoutput[0].t().mm(self._input)
        self.grad_w.add_(d_w)
        return d_in.t()

    def param(self):
        return [[self.weights, self.grad_w], [self.bias, self.grad_b]]


class Tanh(Module):
    def __init__(self):
        self._input = None

    def forward(self, *_input):
        self._input = _input
        return (_input[0].tanh(),)

    def __derivative(self):
        return 4 * (self._input[0].exp() + self._input[0].mul(-1).exp()).pow(-2)

    def backward(self, *gradwrtoutput):
        return self.__derivative() * gradwrtoutput[0]

    def param(self):
        return []

class Sigmoid(Module):
    def __init__(self):
        self._input = None

    def forward(self, *_input):
        self._input = _input
        return (_input[0].sigmoid(),)

    def __derivative(self):
        return self._input[0].sigmoid()*(1-self._input[0].sigmoid())

    def backward(self, *gradwrtoutput):
        return self.__derivative() * gradwrtoutput[0]

    def param(self):
        return []

class ReLU(Module):
    def __init__(self):
        self._input = None
        self.target = None

    def forward(self, *_input):
        self._input = _input
        return (torch.max(torch.tensor(0.0), _input[0]), )

    def _derivative(self):
        t = self._input[0].clone()
        t[t<=0] = 0
        t[t>0] = 1
        return t

    def backward(self, *gradwrtoutput):
        return gradwrtoutput[0] * self._derivative()

    def param(self):
        return[]

class LossMSE(Module):
    def __init__(self):
        self._input = None
        self.target = None

    def forward(self, *_input):
        self._input = _input[0]
        self.target = _input[1]
        res = ((self._input - self.target).pow(2).sum())
        return (res, )

    def backward(self, *gradwrtoutput):
        return 2 * (self._input - self.target)

    def param(self):
        return []


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, *_input):
        res = _input
        for module in self.modules:
            res = module.forward(*res)

        return res

    def backward(self, *gradwrtoutput):
        res = gradwrtoutput[0]

        for module in self.modules[::-1]:
            res = module.backward(res)

        return res

    def param(self):
        return [p for module in self.modules for p in module.param()]

    def zero_grad(self):
        for module in self.modules:
            for p in module.param():
                p[1].zero_()
