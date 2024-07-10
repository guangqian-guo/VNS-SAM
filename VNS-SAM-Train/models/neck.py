from turtle import forward
import torch
import torch.nn as nn
import einops
from torch.autograd import Variable
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        # self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

"""
    Joint Attention module (CA + SA)
"""

class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y


class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class AM(nn.Module):
    def __init__(self, channels, lf):
        super(AM, self).__init__()
        self.CA = CA(lf=lf)
        self.SA = SA(channels)

    def forward(self, x):
        x = self.CA(x)
        x = self.SA(x)
        return x


"""
    Low-Frequency Attention Module (LFA)
"""


class RB(nn.Module):
    def __init__(self, channels, lf):
        super(RB, self).__init__()
        self.RB = BasicConv(channels, channels, 3, padding=1, bn=nn.InstanceNorm2d if lf else nn.BatchNorm2d)

    def forward(self, x):
        y = self.RB(x)
        return y + x


class ARB(nn.Module):
    def __init__(self, channels, lf):
        super(ARB, self).__init__()
        self.lf = lf
        self.AM = AM(channels, lf)
        self.RB = RB(channels, lf)
        if self.lf:
            self.mean_conv1 = ConvLayer(1, 16, 1, 1)
            self.mean_conv2 = ConvLayer(16, 16, 3, 1)
            self.mean_conv3 = ConvLayer(16, 1, 1, 1)

            self.std_conv1 = ConvLayer(1, 16, 1, 1)
            self.std_conv2 = ConvLayer(16, 16, 3, 1)
            self.std_conv3 = ConvLayer(16, 1, 1, 1)

    def PONO(self, x, epsilon=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):
        if self.lf:
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))
        y = self.RB(x)
        y = self.AM(y)
        if self.lf:
            return y * std + mean
        return y


"""
    Guidance-based Upsampling
"""

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


"""
    Guidance-based Feature Aggregation Module (GFA)
"""


class AGF(nn.Module):                            # AGF 又包含两个小模块，一个是
    def __init__(self, channels, lf):
        super(AGF, self).__init__()
        self.ARB = ARB(channels, lf)
        # self.GF = GF(r=2, eps=1e-2)

    def forward(self, high_level, low_level):
        N, C, H, W = high_level.size()
        high_level_small = F.interpolate(high_level, size=(int(H / 2), int(W / 2)), mode='bilinear', align_corners=True)
        y = self.ARB(low_level)

        # y = self.GF(high_level_small, low_level, high_level, y)
        
        return y

    
class MFF(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = BasicConv(in_channels=4*channels, out_channels=channels, kernel_size=1, stride=1)
        self.ARB = ARB(channels=channels, lf=False)
    
    def forward(self, f1, f2, f3, f4):
        f = torch.cat([f1, f2, f3, f4], dim=1)
        f = self.conv(f)
        f = self.ARB(f)
        return f
    
    


"""
refer to FEDER.
"""
class AGFG(nn.Module):
    def __init__(self, channels, lf):
        super(AGFG, self).__init__()
        self.GF1 = AGF(channels, lf)   # AGF module 是相邻两层特征融合的模块，也是核心部分，这部分没必要和它完全一致，可以替换。
        self.GF2 = AGF(channels, lf)
        self.GF3 = AGF(channels, lf)

    def forward(self, f1, f2, f3, f4):  # 是一个自下而上的融合的模块

        y = self.GF1(f2, f1)    
        y = self.GF2(f3, y)
        y = self.GF3(f4, y)
        return y



class Aggregator(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels) -> None:
        super().__init__()
        self.down_sample_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.down_sample_layers.append(
                nn.Sequential(
                    BasicConv2d(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                    ),
                    nn.ReLU()
                )
            )
        
        self.DWT = DWT()
        
        #----------------------------------------------
        # self.AGFG_LL = AGFG(inner_channels, True)
        # self.AGFG_HH = AGFG(inner_channels, False)
        #----------------------------------------------
        
        self.MFF_LL = MFF(inner_channels)
        self.MFF_HH = MFF(inner_channels)
        
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll = nn.Conv2d(in_channels=inner_channels + inner_channels // 4, out_channels=inner_channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=inner_channels + inner_channels // 4, out_channels=256, kernel_size=1)
        self.one_conv_llhh = nn.Conv2d(in_channels=352, out_channels=inner_channels, kernel_size=1)

        
        

    def aggregate(self, LL, HH, f1, f4):
         # 高频信息
        HH_up = self.dePixelShuffle(HH)       # 这个是什么意思？？？
        f1_HH = torch.cat([HH_up, f1], dim=1)  # cat

        f1_HH = self.one_conv_f1_hh(f1_HH)    # 一个卷积层

        # 低频信息
        LL_up = self.dePixelShuffle(LL)
        f4_LL = torch.cat([LL_up, f4], dim=1) 
        
        f4_LL = self.one_conv_f4_ll(f4_LL) 

        # fusion high and low for mask feature
        LLHH =torch.cat([f1_HH, f4_LL], dim=1)
        LLHH = self.one_conv_llhh(LLHH)
        
        return f1_HH, LLHH
    

    def forward(self, inputs):
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inputs)]
        f1, f2, f3, f4 = inner_states

        wf1 = self.DWT(f1)   # 变换
        wf2 = self.DWT(f2)
        wf3 = self.DWT(f3)
        wf4 = self.DWT(f4)
        
        LL = self.MFF_LL(wf4[0], wf3[0], wf2[0], wf1[0])
        HH = self.MFF_HH(wf4[3], wf3[3], wf2[3], wf1[3])

        HH_feat, LLHH_feat= self.aggregate(LL, HH, f1, f4)
        
        return LLHH_feat, HH_feat


class DecoupleHead(nn.Module):
    def __init__(self, inner_channels, out_channels) -> None:
        super().__init__()
        # edge layer        
        self.edge_pred_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
        )

        self.hqfeat_head = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
            nn.ReLU()
        )

        self.up_sample_layers = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm2d(out_channels),
                nn.GELU(), 
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm2d(out_channels),
                nn.GELU(), 
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            )
    
    
    def forward(self, LLHH, HH):
        edge_feat = self.edge_pred_head(HH)
        
        img_feat = self.hqfeat_head(LLHH)
        img_feat = self.up_sample_layers(img_feat)
        
        return img_feat, edge_feat
    


"""
Main  module
"""
    
class TFD(nn.Module):
    def __init__(
            self,
            in_channels=[1024]*4,
            inner_channels=96,
            out_channels=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.aggregator = Aggregator(self.in_channels, self.inner_channels, self.out_channels)
        self.decouple_head = DecoupleHead(self.inner_channels, self.out_channels)


    def forward(self, inputs):
        inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in range(len(self.in_channels))]
        mask_feat, edge_feat = self.aggregator(inner_states)
        img_feat, edge_feat = self.decouple_head(mask_feat, edge_feat)
        return img_feat, edge_feat



if __name__ == "__main__":
    neck = SAMAggregatorNeck()
    neck.cuda()
    x1 = torch.randn(1, 64, 64, 1024).cuda()
    x2 = torch.randn(1, 64, 64, 1024).cuda()
    x3 = torch.randn(1, 64, 64, 1024).cuda()
    x4 = torch.randn(1, 64, 64, 1024).cuda()
    mask_feat, edge_feat = neck([x1, x2, x3, x4])
    print(mask_feat.shape, edge_feat.shape)
    
    
    
    