import dlc_practical_prologue as prologue
import torch
from torch import nn
from torch.nn import functional as F
from model import *

def train_model(model, train_input, train_target, train_classes, epochs=25, mini_batch_size=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))

            # If model not binary compute main loss against class
            if not model.binary:
                loss = criterion(output[0], train_classes.narrow(0, b, mini_batch_size)[:,0])
                loss += criterion(output[1], train_classes.narrow(0, b, mini_batch_size)[:,1])

            # else compute against main target
            else:
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            # Compute auxiliary loss against class information
            if model.aux:
                left, right = model.get_subnetwork_output()
                classes = train_classes.narrow(0, b, mini_batch_size)
                loss += criterion(left, classes[:, 0])
                loss += criterion(right, classes[:, 1])

            loss.backward()
            optimizer.step()


def compute_accuracy(inp, targ, model):
    if model.binary:
        out = model(inp)
        pred = torch.argmax(out, dim=1)
        errors = torch.abs(pred - targ).sum().item()

    else:
        out_1, out_2 = model(inp)
        pred_1, pred_2 = torch.argmax(out_1, dim=1), torch.argmax(out_2, dim=1)

        difference = pred_2 - pred_1
        difference[difference >= 0] = 1
        difference[difference < 0] = 0

        errors = torch.sum(torch.abs(difference - targ)).item()

    return 100 * (1 - errors/inp.shape[0])

def get_results(n_pairs=1000, rounds=10):
    results = []
    model_params = [[False, False, True],
                    [True, False, True],
                    [False, True, True],
                    [True, True, True],

                    [False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [True, True, False]]

    for p in model_params:
        model = Model(weight_sharing=p[0], aux=p[1], binary=p[2])
        accuracies = []
        model_id = model_params.index(p)+1

        print("Model {}:".format(model_id))

        for i in range(rounds):

            # generate train/test data
            train_input, train_target, train_classes, test_input, test_target, test_classes = \
                prologue.generate_pair_sets(n_pairs)

            # normalize inputs
            train_input = train_input.sub_(torch.mean(train_input)).div_(torch.std(train_input))
            test_input = test_input.sub_(torch.mean(test_input)).div_(torch.std(test_input))

            # train model
            train_model(model, train_input, train_target, train_classes)

            # compute test accuracy
            accuracy = compute_accuracy(test_input, test_target, model)
            accuracies.append(accuracy)

        accuracy_avg = round(sum(accuracies)/rounds,2)
        accuracy_stdev = round((sum([(x-accuracy_avg)**2 for x in accuracies])/(rounds-1))**0.5,2)
        results.append([model_id, accuracy_avg, accuracy_stdev])

        print("Avg test acc %: {}".format(accuracy_avg))
        print("Stdev test acc %: {}".format(accuracy_stdev))
        print('-'*30)

    return results

test_results = get_results()
