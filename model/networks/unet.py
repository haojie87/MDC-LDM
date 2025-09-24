
import torch
import torch.nn as nn
import sys
root_file = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(root_file)
from model.networks.base import BaseNet
from model.modules.attention import SelfAttention
from model.modules.block import DownBlock, UpBlock
from model.modules.conv import DoubleConv


class UNet(BaseNet):
    """
    UNet
    """

    def __init__(self, **kwargs):
        """
        Initialize the UNet network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super(UNet, self).__init__(**kwargs)

        # channel: 3 -> 64
        # size: size
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

        # channel: 64 -> 128
        # size: size / 2
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa1 = SelfAttention(channels=self.channel[2], size=self.image_size_list[1], act=self.act)
        # channel: 128 -> 256
        # size: size / 4
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = SelfAttention(channels=self.channel[3], size=self.image_size_list[2], act=self.act)
        # channel: 256 -> 256
        # size: size / 8
        self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=self.channel[3], size=self.image_size_list[3], act=self.act)

        # channel: 256 -> 512
        # size: size / 8
        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 512
        # size: size / 8
        self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 256
        # size: size / 8
        self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=self.channel[2], size=self.image_size_list[2], act=self.act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=self.channel[1], size=self.image_size_list[1], act=self.act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=self.channel[1], size=self.image_size_list[0], act=self.act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        time = self.encode_time_with_label(time=time, y=None)

        x1 = self.inc(x)
        y1 = self.inc(y)
        x2, y2 = self.down1(x1, time, y1)
        x2_sa = self.sa1(x2)
        x3, y3 = self.down2(x2_sa, time, y2)
        x3_sa = self.sa2(x3)
        x4, y4 = self.down3(x3_sa, time, y3)
        x4_sa = self.sa3(x4)

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)
        y4 = self.bot1(y4)
        y4 = self.bot2(y4)
        y4 = self.bot3(y4)

        up1_out, y5 = self.up1(bot3_out, x3_sa, time, y4, y3)
        up1_sa_out = self.sa4(up1_out)
        up2_out, y6 = self.up2(up1_sa_out, x2_sa, time, y5, y2)
        up2_sa_out = self.sa5(up2_out)
        up3_out, y7 = self.up3(up2_sa_out, x1, time, y6, y1)
        up3_sa_out = self.sa6(up3_out)
        output = self.outc(up3_sa_out)
        return output


if __name__ == "__main__":
    # Unconditional
    net = UNet(device="cpu", image_size=(64, 64))
    net = UNet(in_channel=8, out_channel=8, channel=None, time_channel=256,
                 num_classes=None, image_size=None, device="cpu", act="silu")
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(10, 8, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    # y = x.new_tensor([1] * x.shape[0]).long()
    y = torch.randn(10, 8, 64, 64)
    print(net(x, t, y).shape)
    # print(net(x, t, y).shape)
