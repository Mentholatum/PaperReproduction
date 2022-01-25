import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Dense conv. block
    """
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.Conv(x)
        return x

class UNetPP(nn.Module):
    """
    A Nested U-Net Architecture(UNet++)
    """
    def __init__(self, args, ch_in, ch_out):
        super(UNetPP, self).__init__()

        self.args = args

        nb_filter  = [32, 64, 128, 256, 512]

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Conv0_0 = DoubleConv(ch_in, nb_filter[0])
        self.Conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.Conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.Conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.Conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # 先前的feature map会累积并到达当前节点的原因是因为在每个skip-pathway上使用了密集卷积块
        self.Conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.Conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.Conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.Conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.Conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.Conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.Conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.Conv0_3 = DoubleConv(nb_filter[2] * 3 + nb_filter[1], nb_filter[0])
        self.Conv1_3 = DoubleConv(nb_filter[2] * 3 + nb_filter[2], nb_filter[1])

        self.Conv0_4 = DoubleConv(nb_filter[2] * 3 + nb_filter[1], nb_filter[0])

        self.Sigmoid = nn.Sigmoid()

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], ch_out, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], ch_out, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], ch_out, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], ch_out, kernel_size=1)
        else:
            self.final1 = nn.Conv2d(nb_filter[0], ch_out, kernel_size=1)

    def forward(self,x):
        x0_0 = self.Conv0_0(x)
        x1_0 = self.Conv1_0(self.MaxPool(x0_0))
        x0_1 = self.Conv0_1(torch.cat([x0_0, self.Up(x1_0)], dim=1))

        x2_0 = self.Conv2_0(self.MaxPool(x1_0))
        x1_1 = self.Conv1_1(torch.cat([x1_0, self.Up(x2_0)], dim=1))
        x0_2 = self.Conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], dim=1))

        x3_0 = self.Conv3_0(self.MaxPool(x2_0))
        x2_1 = self.Conv2_1(torch.cat([x2_0, self.Up(x3_0)], dim=1))
        x1_2 = self.Conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], dim=1))
        x0_3 = self.Conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], dim=1))

        x4_0 = self.Conv4_0(self.MaxPool(x3_0))
        x3_1 = self.Conv3_1(torch.cat([x3_0, self.Up(x4_0)], dim=1))
        x2_2 = self.Conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], dim=1))
        x1_3 = self.Conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], dim=1))
        x0_4 = self.Conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], dim=1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output1 = self.Sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.Sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.Sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.Sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.Sigmoid(output)
            return output






