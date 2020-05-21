import torch

class Optimizer(object):

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, params, lr):
        super(SGD, self).__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p[0] -= self.lr * p[1]

class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def step(self):
        if self.m is None:
            self.m = [torch.zeros(param[0].shape) for param in self.params]
        if self.v is None:
            self.v = [torch.zeros(param[0].shape) for param in self.params]

        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param[1]
            self.v[i] = self.beta2 * self.v[i]  + (1 - self.beta2) * param[1] * param[1]
            m_corr = self.m[i] / (1 - self.beta1)
            v_corr = self.v[i] / (1 - self.beta2)
            param[0] -= self.lr * m_corr/((v_corr.sqrt()) + self.epsilon)
