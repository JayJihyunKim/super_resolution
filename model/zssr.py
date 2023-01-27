import torch.nn as nn


class ZSSR(nn.Module):
    def __init__(self):
        super(ZSSR, self).__init__()
        self.head = conv2d(3,64, activation='relu')

        m = [
            conv2d(64,64,activation='relu')
            for _ in range(2,8)
        ]
        self.body = nn.Sequential(*m)
        self.tail = conv2d(64,3)

    def forward(self, x):
        self.input = x
        y = self.head(self.input)
        y = self.body(y)
        y = self.tail(y)
        self.output = self.input + y

        return self.output


def conv2d(in_channel, out_channel, kernel=3, strides=1, activation=None):
    m = []
    pad = nn.ZeroPad2d(1)
    m.append(pad)

    conv = nn.Conv2d(in_channel, out_channel, kernel, strides)
    m.append(conv)
    nn.init.kaiming_normal(conv.weight)
    nn.init.zeros_(conv.bias)

    if activation == 'relu':
        m.append(nn.ReLU(True))

    return nn.Sequential(*m)

