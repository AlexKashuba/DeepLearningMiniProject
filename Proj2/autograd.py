from torch import empty
from torch import set_grad_enabled

# remove before submission
from torch import float64

set_grad_enabled(False)


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

        self.weights = empty(hidden_units, in_features).normal_()
        self.grad_w = empty(self.weights.size()).zero_()

        self.bias = empty(hidden_units).normal_()
        self.grad_b = empty(self.bias.size()).zero_()

    def forward(self, *_input):
        if self.name:
            print("Layer: {}".format(self.name))
        self._input = _input[0]
        # print("Forward input: {}".format(_input))
        res = (self.weights.mv(self._input) + self.bias,)
        # print(res)
        return res

    def backward(self, *gradwrtoutput):
        # print("weights: {}".format(self.weights))
        self.grad_b.add_(gradwrtoutput[0])
        d_in = self.weights.t().mv(gradwrtoutput[0])
        d_w = gradwrtoutput[0].view(-1, 1).mm(self._input.view(1, -1))
        self.grad_w.add_(d_w)
        return d_in

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
        return gradwrtoutput[0] * self.__derivative()

    def param(self):
        return []

# class ReLU(Module):
#     def __init__(self)
#         self._input = None

#     def forward(self, *_input):
#         self._input = _input
        
class LossMSE(Module):
    def __init__(self):
        self._input = None
        self.target = None

    def forward(self, *_input):
        self._input = _input[0]
        self.target = _input[1]
        res = ((self._input - self.target).pow(2).sum())
        # print(res)
        return (res, )

    def backward(self, *gradwrtoutput):
        return 2 * (self._input - self.target)  # * gradwrtoutput[0]

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


if __name__ == "__main__":
    from math import pi
    from torch import tensor  # remove before submission?

    CENTER_X, CENTER_Y = 0.5, 0.5
    RADIUS_SQUARED = 1/(2*pi)
    N_POINTS = 1000

    def inside_circle(point):
        res = [0, 0]
        if ((point[0] - CENTER_X).pow(2) + (point[1] - CENTER_Y).pow(2)) <= RADIUS_SQUARED:
            res[1] = 1
        else:
            res[0] = 1

        return res

    def gen_data():
        points = empty(1000, 2).uniform_()
        labels = []
        for point in points:
            labels.append(inside_circle(point))

        labels = tensor(labels)
        return points, labels

    def correct_prediction(output, label):
        _, predicted = output[0].max(0)
        _, real = label.max(0)
        return predicted == real

    train_input, train_labels = gen_data()
    test_input, test_labels = gen_data()

    model = Sequential(
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 2))

    loss = LossMSE()

    eta = 1e-6
    epochs = 500

    for e in range(epochs):
        sum_loss = 0
        errors = 0

        for i in range(N_POINTS):
            output = model.forward(train_input[i])
            if not correct_prediction(output, train_labels[i]):
                errors += 1
            loss_val = loss.forward(*output, train_labels[i])[0]
            model.backward(loss.backward())
            sum_loss = sum_loss + loss_val

            for p in model.param():
                p[0] -= eta * p[1]

            model.zero_grad()

        print("Epoch: {}, loss: {}, acc: {}".format(
            e, sum_loss, 1 - errors/N_POINTS))

    sum_loss = 0
    errors = 0
    for i in range(N_POINTS):
        output = model.forward(test_input[i], test_labels[i])
        loss_val = loss.forward(*output, test_labels[i])[0]
        if not correct_prediction(output, test_labels[i]):
                errors += 1
        sum_loss = sum_loss + loss_val

    print("loss: {}, acc: {}".format(sum_loss, 1 - errors/N_POINTS))
