import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, squeeze=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, squeeze, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        # Squeeze
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = self.conv1(out)
        out = self.relu(out)
        # Excitation
        out = self.conv2(out)
        out = self.sigmoid(out)
        return identity * out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in//4, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(c_in//4)
        self.conv2 = nn.Conv2d(c_in//4, c_in//4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(c_in//4)
        self.conv3 = nn.Conv2d(c_in//4, c_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        
        self.downsample = self._downsample(c_in, c_out, stride)
        self.stride = stride
        self.seblock = SEBlock(c_out, c_out//12+1)

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)

        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)

        out = self.relu(out)
        out = self.bn3(out)
        out = self.conv3(out)
        
        identity = self.downsample(x)

        out = self.seblock(out)

        out += identity

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
        self.layers = self._make_layer(block, num_channels, num_channels, 10, 1)
        self.pi = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(bs*bs*2, action_size),
            nn.LogSoftmax(-1)
        )
        self.v = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(bs*bs, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.view(-1, 12, self.board_size, self.board_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layers(x)
        x = self.relu(x)
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
    model = DualNet(7, 50, 128)
    model = model.to(device)
    # model.apply(initialize_weights)
    dummy = torch.rand([3, 12, 7, 7]).to(device)
    traced_net = torch.jit.trace(model, dummy)
    # print(traced_net)
    # for param in model.parameters():
    #     print(param)
    traced_net.save("dualnet7x7_se.pt")
    print("saved model")
