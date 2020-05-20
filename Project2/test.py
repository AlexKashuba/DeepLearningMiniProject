from torch import empty
from torch import set_grad_enabled

# remove before submission
from torch import float64
import torch
set_grad_enabled(False)
from modules import *
from helpers import *


if __name__ == "__main__":
    from math import pi
    from math import sqrt
    from torch import tensor

    def correct_prediction(output, label):
        _, predicted = output[0].max(1)
        _, real = label.max(1)
        return (predicted - real).abs().sum()

    eta = 1e-3
    epochs = 500
    mini_batch_size = 50
    N_POINTS = 1000

    train_input, train_labels = generate_circle_data(N_POINTS)
    test_input, test_labels = generate_circle_data(N_POINTS)

    model = Sequential(
        Linear(2, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 2),
        ReLU())


    loss = LossMSE()


    for e in range(epochs):
        sum_loss = 0
        errors = 0

        for i in range(0, N_POINTS, mini_batch_size):
            output = model.forward(train_input.narrow(0, i, mini_batch_size))
            errors += correct_prediction(output, train_labels.narrow(0, i, mini_batch_size)).item()
            loss_val = loss.forward(*output, train_labels.narrow(0, i, mini_batch_size))[0]
            model.backward(loss.backward())
            sum_loss = sum_loss + loss_val

            for p in model.param():
                p[0] -= eta * p[1]

            model.zero_grad()

        print("Epoch: {}, loss: {}, acc: {}".format(
            e, sum_loss, 1 - float(errors)/N_POINTS))

    sum_loss = 0
    errors = 0
    mini_batch_sizet = 50
    for i in range(0, N_POINTS, mini_batch_size):
        output = model.forward(test_input.narrow(0, i, mini_batch_size), test_labels[i])
        loss_val = loss.forward(*output, test_labels.narrow(0, i, mini_batch_size))[0]
        errors += correct_prediction(output, test_labels.narrow(0, i, mini_batch_size))
        sum_loss = sum_loss + loss_val

    print("test loss: {}, test acc: {}".format(sum_loss, 1 - float(errors)/N_POINTS))
