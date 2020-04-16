import dlc_practical_prologue as prologue
import torch
from torch import nn
from torch.nn import functional as F

N_PAIRS = 1000
MINI_BATCH_SIZE = 100  
EPOCHS = 25
LEFT = 0
RIGHT = 1

def train_model(model, train_input, train_target, train_classes=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(EPOCHS):
        sum_loss = 0
        for b in range(0, train_input.size(0), MINI_BATCH_SIZE):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, MINI_BATCH_SIZE))
            loss = criterion(output, train_target.narrow(0, b, MINI_BATCH_SIZE))
            sum_loss = sum_loss + loss.item()
            
            if train_classes is not None:
                left, right = model.get_subnetwork_output()
                classes = train_classes.narrow(0, b, MINI_BATCH_SIZE)
                loss += criterion(left, classes[:, LEFT])
                loss += criterion(right, classes[:, RIGHT])

            loss.backward()
            optimizer.step()

        print(e, sum_loss)

def compute_nb_errors(model, _input, target):
    nb_erros = 0
    for b in range(0, _input.size(0), MINI_BATCH_SIZE):
        output_res = model(_input.narrow(0, b, MINI_BATCH_SIZE))
        _, predicted = output_res.max(1)
        for i in range(MINI_BATCH_SIZE):
            real = target[b+i]
            if real != predicted[i]:
                nb_erros += 1

    return nb_erros


class DigitSubnetwork(nn.Module):

    def __init__(self, nb_hidden=200):
        super(DigitSubnetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(4 * 4 * 64, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)
        x = F.relu(self.fc1(x.view(-1, 4 * 4 * 64)))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x

class Model(nn.Module):
    def __init__(self, weight_sharing=False):
        super(Model, self).__init__()
        self.left = DigitSubnetwork()
        self.weight_sharing = weight_sharing
        if not weight_sharing:
            self.right = DigitSubnetwork()
        self.fc = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)
        self.left_out = None
        self.right_out = None

    def get_subnetwork_output(self):
        return (self.left_out, self.right_out)

    def forward(self, x):
        left = self.left(x[:, LEFT, :, :].unsqueeze(1))
        if not self.weight_sharing:
            right = self.right(x[:, RIGHT, :, :].unsqueeze(1))
        else:
            right = self.left(x[:, RIGHT, :, :].unsqueeze(1))

        self.left_out = left
        self.right_out = right

        x = torch.cat((left, right), dim=1)
        #print(x.shape)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def scatter(_input):
    n = _input.shape[0]
    z = torch.zeros(n, 2)
    z.scatter_(1, _input.view(-1, 1), torch.ones(n, 1))
    return z


if __name__ == "__main__":
    model = Model(weight_sharing=False)
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(N_PAIRS)

    train_model(model, train_input, train_target, train_classes)
    errs = compute_nb_errors(model, test_input, test_target)
    print("Error: {}".format(errs/test_input.shape[0]))
