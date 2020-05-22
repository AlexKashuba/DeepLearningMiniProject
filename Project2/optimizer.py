from torch import empty
from math import pow


class Optimizer(object):
    """Base class for all optimizers"""

    def step(self):
        """Perform a single optimization step"""
        raise NotImplementedError


class SGD(Optimizer):
    """A simple Stochastic Gradient Descent Optimizer"""

    def __init__(self, params, lr):
        """Arguments:
            params: tensors to be optimized
            lr: (float) learning rate
        """
        super(SGD, self).__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p[0] -= self.lr * p[1]


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Arguments:
            params: tensors to be optimized
            lr: (float) learning rate
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [empty(param[0].shape).zero_() for param in self.params]
        self.v = [empty(param[0].shape).zero_() for param in self.params]
        self.t = 1

    def step(self):
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param[1]
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * (param[1]**2)
                
            m_hat = self.m[i] / (1 - pow(self.beta1, self.t))
            v_hat = self.v[i] / (1 - pow(self.beta2, self.t))
            param[0] -= self.lr * m_hat/((v_hat.sqrt()) + self.epsilon)

        self.t += 1
