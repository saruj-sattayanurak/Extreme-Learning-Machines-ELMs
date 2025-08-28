import torch
import torch.nn as nn
import torch.nn.functional as F

class MyExtremeLearningMachine(torch.nn.Module):
    def __init__(self, hidden_size=1000):
        super().__init__()
        self.input_size = 3 * 32 * 32 # CIFAR-10 are of size 3x32x32
        self.hidden_size = hidden_size
        self.output_size = 10 # CIFAR-10 have 10 classes

        # fix value, not trainable
        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=True)

        # trainable
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size, bias=True)

        # ramdomly assign value
        self.initialise_fixed_layers()

    def initialise_fixed_layers(self):
        with torch.no_grad():
            nn.init.normal_(self.input_to_hidden.weight, mean=0, std=1)
            nn.init.normal_(self.input_to_hidden.bias, mean=0, std=1)
            self.input_to_hidden.weight.requires_grad = False
            self.input_to_hidden.bias.requires_grad = False

    def forward(self, x):
        x = torch.flatten(x, 1)
        hidden = F.relu(self.input_to_hidden(x))
        output = self.hidden_to_output(hidden)

        return output
