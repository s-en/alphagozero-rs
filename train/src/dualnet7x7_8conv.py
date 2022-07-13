
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DualNet(nn.Module):
    def __init__(self, bsize, action_size, num_channels):
        # game params
        self.bsize = bsize
        self.num_channels = num_channels
        self.action_size = action_size

        super(DualNet, self).__init__()
        self.conv1 = nn.Conv2d(12, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv7 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)
        self.bn5 = nn.BatchNorm2d(num_channels)
        self.bn6 = nn.BatchNorm2d(num_channels)
        self.bn7 = nn.BatchNorm2d(num_channels)
        self.bn8 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels*(self.bsize-4)*(self.bsize-4), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.action_size)

        self.fc4 = nn.Linear(256, 1)

        self.dummy = nn.Linear(8*3*3, 256)
        # self._initialize_weights()

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 12, self.bsize, self.bsize)                # batch_size x 1 x board_x x board_y
        # print(s.shape)
        # s = self.dummy(s)
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn5(self.conv5(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn6(self.conv6(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn7(self.conv7(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn8(self.conv8(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.bsize-4)*(self.bsize-4))
        # s = F.relu(self.fc_bn1(self.fc1(s)))  # batch_size x 1024
        # s = F.relu(self.fc_bn2(self.fc2(s)))  # batch_size x 512
        s = self.fc1(s)  # batch_size x 1024
        s = self.fc2(s)  # batch_size x 512
        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1
        return F.log_softmax(pi, dim=-1).exp(), torch.tanh(v)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    bsize = 7
    model = DualNet(bsize, bsize*bsize+1, 32)
    model = model.to(device)
    dummy = torch.rand([8, 12, bsize, bsize]).to(device)
    traced_net = torch.jit.trace(model, dummy)
    traced_net.save("dualnet7x7_8conv.pt")
    print("saved model")