import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_shape, no_layers, hidden_size=512):
        super(FFN, self).__init__()
        # container
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hidden_size)
        self.relu1 = nn.ReLU()
        self.layer_container = nn.ModuleList()
        for _ in range(no_layers):
            self.layer_container.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.layer_container.append(nn.ReLU())
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        x = self.relu1(self.input_layer(x))
        for layer in self.layer_container:
            x = layer(x)
        out = self.output_layer(x)
        return out
