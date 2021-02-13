import torch
import math


class HulaNet(torch.nn.Module):
    def __init__(self, conv1, conv2, out):
        super(HulaNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 5)
        self.conv2 = torch.nn.Conv2d(4, 16, 3)
        self.pool2 = torch.nn.AvgPool2d((2, 2))
        self.full1 = torch.nn.Linear(16 * (((1080 // 120 * 10) - 4 - 2) // 2) * (((1920 // 120 * 10) - 4 - 2) // 2),
                                     16 * 12 * 2)
        self.full2 = torch.nn.Linear(16 * 12 * 2, out)
        # init weights
        torch.nn.init.xavier_normal_(self.conv1.weight.data)
        torch.nn.init.xavier_normal_(self.conv2.weight.data)
        torch.nn.init.xavier_normal_(self.full1.weight.data)
        torch.nn.init.xavier_normal_(self.full2.weight.data)

    #   self.conv1.weight = torch.nn.init.xavier_normal_(torch.ones(5, 5))

    def forward(self, x):
        x = torch.nn.functional.celu(self.conv1(x))
        x = torch.nn.functional.celu(self.conv2(x))
        x = self.pool2(x)
        x = x.view((-1, 16 * (((1080 // 120 * 10) - 4 - 2) // 2) * (((1920 // 120 * 10) - 4 - 2) // 2)))
        x = self.full1(x)
        x = torch.nn.functional.celu(x)
        x = self.full2(x)
        x = torch.nn.functional.softmax(x,dim=1)
        return x
