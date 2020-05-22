from torch import set_grad_enabled
from torch import manual_seed

from optimizer import *
from helpers import *
from modules import *

set_grad_enabled(False)

LEARNING_RATE = 1e-3
EPOCHS = 100
MINI_BATCH_SIZE = 50
N_POINTS = 1000


def train_model(model, train_input, train_labels, test_input, test_labels, optimizer, shouldPrint=True):
    loss = LossMSE()

    for e in range(EPOCHS):
        sum_loss = 0
        errors = 0

        for i in range(0, N_POINTS, MINI_BATCH_SIZE):
            output = model.forward(train_input.narrow(0, i, MINI_BATCH_SIZE))
            errors += nb_errors(output, train_labels.narrow(0,
                                                            i, MINI_BATCH_SIZE)).item()
            loss_val = loss.forward(
                *output, train_labels.narrow(0, i, MINI_BATCH_SIZE))[0]
            model.backward(loss.backward())
            sum_loss = sum_loss + loss_val
            optimizer.step()

            model.zero_grad()
        if shouldPrint:
            print("Epoch: {}, loss: {}, acc: {}".format(
                e, sum_loss, 1 - float(errors)/N_POINTS))

    sum_loss = 0
    errors = 0

    for i in range(0, N_POINTS, MINI_BATCH_SIZE):
        output = model.forward(test_input.narrow(
            0, i, MINI_BATCH_SIZE), test_labels[i])
        loss_val = loss.forward(
            *output, test_labels.narrow(0, i, MINI_BATCH_SIZE))[0]
        errors += nb_errors(output, test_labels.narrow(0, i, MINI_BATCH_SIZE))
        sum_loss = sum_loss + loss_val

    test_acc = 1 - float(errors)/N_POINTS

    if shouldPrint:
        print("######################################################\n")
        print("Test loss: {}, test acc: {}".format(
            sum_loss, test_acc))

    return test_acc


def experiments():
    ROUNDS = 10
    train = [
        generate_circle_data(N_POINTS) for i in range(ROUNDS)]
    test = [
        generate_circle_data(N_POINTS) for i in range(ROUNDS)]

    optimizers = {"Adam": Adam, "SGD": SGD}

    models_to_test = {opt: {
        "relu": Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 2),
            ReLU()),

        "tanh": Sequential(
            Linear(2, 25),
            Tanh(),
            Linear(25, 25),
            Tanh(),
            Linear(25, 25),
            Tanh(),
            Linear(25, 2),
            Tanh()),

        "sigmoid": Sequential(
            Linear(2, 25),
            Sigmoid(),
            Linear(25, 25),
            Sigmoid(),
            Linear(25, 25),
            Sigmoid(),
            Linear(25, 2),
            Sigmoid()),
    } for opt in optimizers.keys()}

    from math import sqrt, pow 
    
    for optimizer_name, models in models_to_test.items():
        for model_name, model in models.items():
            acc = [
                train_model(model, train[i][0], train[i][1], test[i][0], test[i][1], optimizers[optimizer_name](
                    model.param(), lr=LEARNING_RATE), shouldPrint=False)
                for i in range(ROUNDS)
            ]
            accuracy_avg = sum(acc)/ROUNDS
            stddev_acc = sqrt(sum([pow((x-accuracy_avg), 2) for x in acc])/(ROUNDS - 1))
            print(
                ", ".join([model_name, optimizer_name, str(round(accuracy_avg, 3)), str(round(stddev_acc, 3))]))


if __name__ == "__main__":
    manual_seed(5)

    # experiments()

    model = Sequential(
        Linear(2, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 25),
        ReLU(),
        Linear(25, 2),
        ReLU())

    train_input, train_labels = generate_circle_data(N_POINTS)
    test_input, test_labels = generate_circle_data(N_POINTS)
    optimizer = Adam(model.param(), lr=LEARNING_RATE)
    train_model(model, train_input, train_labels,
                test_input, test_labels, optimizer)
