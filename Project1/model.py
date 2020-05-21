import torch
from torch import nn
from torch.nn import functional as F

class DigitSubnetwork(nn.Module):

    def __init__(self, aux=False, nb_hidden=200):
        super(DigitSubnetwork, self).__init__()
        self.aux = aux
        self.aux_out = None
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(4 * 4 * 16, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

        if self.aux:
            self.fcaux = nn.Linear(6 * 6 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2))

        if self.aux:
            self.aux_out = self.fcaux(x.view(-1, 6 * 6 * 8))

        x = self.conv2(x)

        x = F.relu(x)

        x = F.relu(self.fc1(x.view(-1, 4 * 4 * 16)))
        x = self.fc2(x)

        return x, self.aux_out

class Model(nn.Module):
    def __init__(self, weight_sharing=False, aux=False, binary=True):
        super(Model, self).__init__()
        # compute auxiliary loss from DigitSubnetwork hidden layer if model output not binary
        self.left = DigitSubnetwork(aux and not binary)
        self.weight_sharing = weight_sharing
        self.aux = aux
        self.binary = binary

        if not weight_sharing:
            self.right = DigitSubnetwork(aux and not binary)

        if binary:
            self.fc = nn.Linear(20, 200)
            self.fc2 = nn.Linear(200, 2)

        self.left_out = None
        self.right_out = None

    def get_subnetwork_output(self):
        return (self.left_out, self.right_out)

    def forward(self, x):
        left, left_aux_out = self.left(x[:, 0, :, :].unsqueeze(1))
        if not self.weight_sharing:
            right, right_aux_out = self.right(x[:, 1, :, :].unsqueeze(1))
        else:
            right, right_aux_out = self.left(x[:, 1, :, :].unsqueeze(1))

        self.left_out = left
        self.right_out = right

        if self.aux and not self.binary:
            self.left_out = left_aux_out
            self.right_out = right_aux_out

        if not self.binary:
            return left, right
        else:
            x = torch.cat((left, right), dim=1)
            x = self.fc(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
