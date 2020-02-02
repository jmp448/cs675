""" Model classes defined here! """

import torch
import torch.nn.functional as F


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 10, kernel_size=n2_kern, stride=2)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)
        x = x.view(-1, 10)
        return x


class BestNN(torch.nn.Module):
    def __init__(self, n1_chan, n2_chan, n3_chan):
        super(BestNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=10, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(n1_chan, n2_chan, kernel_size=4)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(n2_chan, n3_chan, kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(9*n3_chan, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.relu(x)
        x = x.view(-1, 10)
        return x
