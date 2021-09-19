import torch
from torch import Tensor
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample = self._downsample(c_in, c_out, stride)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def _downsample(self, c_in, c_out, stride=1):
        if stride != 1 or c_in != c_out:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            return nn.Sequential()

class DualNet(nn.Module):

    def __init__(self, board_size, action_size, num_channels):
        super().__init__()
        block = BasicBlock
        bs = board_size
        self.board_size = board_size
        self.conv1 = nn.Conv2d(12, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self._make_layer(block, num_channels, num_channels, 8, 1)
        self.pi = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1, 1, 0, bias=False), # [1, 32, 5, 5]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(bs*bs*32, action_size),
            nn.LogSoftmax(-1)
        )
        self.v = nn.Sequential(
            nn.Conv2d(num_channels, 3, 1, 1, 0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(bs*bs*3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.view(-1, 12, self.board_size, self.board_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        pi = self.pi(x).exp()
        v = self.v(x).tanh()
        res = torch.cat((pi, v), 1)
        print(res.shape)

        return res
    
    def _make_layer(self, block, c_in, c_out, blocks, stride=1):
        layers = []
        layers.append(block(c_in, c_out, stride))

        for _ in range(1, blocks):
            layers.append(block(c_out, c_out, 1))

        return nn.Sequential(*layers)

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = DualNet(5, 26, 32)
    model = model.to(device)
    # model.apply(initialize_weights)
    dummy = torch.rand([16, 12, 5, 5]).to(device)
    traced_net = torch.jit.trace(model, dummy)
    # print(traced_net)
    # for param in model.parameters():
    #     print(param)
    traced_net.save("dualnet.pt")
    print("saved model")
