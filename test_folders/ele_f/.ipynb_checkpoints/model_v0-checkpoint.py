import torch
import torch.nn as nn


class feed_forward(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.network = nn.ModuleList()
        
        for i in range(len(self.args)-2):
            self.network.append(
                nn.Linear(
                    self.args[i], self.args[i+1]))
            if i < len(self.args)-3:
                self.network.append(nn.SiLU())
        
    def forward(self, data):
        #data = data.view(data.shape[0], -1)
        #data = data.view(-1)             
        output_info = [data, ]
        input_data = data

        for i, layer in enumerate(self.network):
            output_data = layer(input_data)
            output_info.append(output_data)
            input_data = output_data

        return output_info