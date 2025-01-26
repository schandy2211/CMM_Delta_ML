import torch
import torch.nn as nn


class feed_forward(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.network = nn.ModuleList()
        
        for i in range(len(self.args)-1):
            self.network.append(
                nn.Linear(
                    self.args[i], self.args[i+1]))
            if i < len(self.args)-2:
                self.network.append(nn.SiLU())
        
    def forward(self, data):
        input_data = data

        for i, layer in enumerate(self.network):
            output_data = layer(input_data)
            input_data = output_data

        return output_data
