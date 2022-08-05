
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    """
    Bottleneckを使用したresidual blockクラス
    """
    def __init__(self, indim, outdim,):
        super(Bottleneck, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=1, padding=0)
        
        dim_inter = int(outdim / 4)
        self.conv1 = nn.Conv2d(indim, dim_inter , 1)
        self.bn1 = nn.BatchNorm2d(dim_inter)
        self.conv2 = nn.Conv2d(dim_inter, dim_inter, 3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_inter)
        self.conv3 = nn.Conv2d(dim_inter, outdim, 1)
        self.bn3 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out

class DualNet(nn.Module):
    def __init__(self, bsize, action_size, num_channels):
        # game params
        self.bsize = bsize
        self.num_channels = num_channels
        self.action_size = action_size

        super(DualNet, self).__init__()
        self.block1 = Bottleneck(12, num_channels)
        self.block2 = Bottleneck(num_channels, num_channels)
        self.block3 = Bottleneck(num_channels, num_channels)
        self.block4 = Bottleneck(num_channels, num_channels)
        self.block5 = Bottleneck(num_channels, num_channels)
        self.block6 = Bottleneck(num_channels, num_channels)
        self.block7 = Bottleneck(num_channels, num_channels)
        self.block8 = Bottleneck(num_channels, num_channels)

        self.conv_p = nn.Conv2d(num_channels, 2, 1)
        self.bn_p = nn.BatchNorm2d(2)
        self.fc_p = nn.Linear(bsize*bsize*2, self.action_size)

        self.conv_v = nn.Conv2d(num_channels, 1, 1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v = nn.Linear(bsize*bsize, 128)
        self.fc_v2 = nn.Linear(128, 1)
        # self._initialize_weights()

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(-1, 12, self.bsize, self.bsize) # 8, 12, 7, 7
        # print(s.shape)
        # s = self.dummy(s)
        s = self.block1(s)                          # 8, 128, 7, 7
        s = self.block2(s)                          # 8, 128, 7, 7
        s = self.block3(s)                          # 8, 128, 7, 7
        s = self.block4(s)                          # 8, 128, 7, 7
        s = self.block5(s)
        s = self.block6(s)
        s = self.block7(s)
        s = self.block8(s)
        # s = s.view(-1, self.num_channels*4*bsize*bsize)
        # print(s.size())
        print(s.size())

        pi = self.bn_p(self.conv_p(s)).relu()
        print(pi.size())
        pi = pi.view(-1,self.bsize*self.bsize*2)
        print(pi.size())
        pi = self.fc_p(pi)
        print(pi.size())
        
        v = self.bn_v(self.conv_v(s)).relu()
        v = self.fc_v(v.view(-1,self.bsize*self.bsize))
        v = self.fc_v2(v)
        print(v.size())
        return F.log_softmax(pi, dim=-1).exp(), torch.tanh(v)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    bsize = 7
    model = DualNet(bsize, bsize*bsize+1, 128)
    model = model.to(device)
    dummy = torch.rand([8, 12, bsize, bsize]).to(device)
    traced_net = torch.jit.trace(model, dummy)
    traced_net.save("dualnet7x7_bn.pt")
    print("saved model")