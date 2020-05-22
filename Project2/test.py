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


def train_model(model, train_input, train_labels, test_input, test_labels, optimizer):
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

    print("######################################################\n")
    test_acc = 1 - float(errors)/N_POINTS
    print("Test loss: {}, test acc: {}".format(
        sum_loss, test_acc))
        
    return test_acc


def experiments():
    models = [
        Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 2),
            ReLU()),
    ]


if __name__ == "__main__":
    manual_seed(5)

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
