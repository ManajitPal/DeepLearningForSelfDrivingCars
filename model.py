import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 24, 5, stride=2, bias=False, padding=1),
#             nn.ELU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(24, 36, 5, stride=2, bias=False),
#             nn.ELU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(36, 48, 5, stride=2, bias=False),
#             nn.ELU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(48, 64, 3, stride=1, bias=False),
#             nn.ELU(),
#             nn.Conv2d(64, 64, 3, stride=1, bias=False),
#             nn.ELU(),
#             nn.MaxPool2d(4, stride=4),
#             nn.Dropout(p=0.25)
#         )
#         self.linear_layers = nn.Sequential(
#             nn.Linear(in_features=64*4*1, out_features=100, bias=False),
#             nn.ELU(),
#             nn.Linear(in_features=100, out_features=50, bias=False),
#             nn.ELU(),
#             nn.Linear(in_features=50, out_features=10, bias=False),
#             nn.ELU(),
#             nn.Linear(in_features=10, out_features=1, bias=False),
#         )
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 16 x 32
            nn.Conv2d(3, 24, 3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2, bias=False),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=48*4*19, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.Linear(in_features=10, out_features=1, bias=False),
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output