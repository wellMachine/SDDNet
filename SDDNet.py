import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from torch import Tensor
from typing import List
from sam2.var import MyVarFeatureEnhance
from typing import Tuple


###############################
# CoordAtt 及其依赖模块定义
###############################

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()
        # kernel_size 只能为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用全局最大池化获得空间显著性图
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class NonLocalAttention(nn.Module):
    """Non-Local Attention module for capturing global dependencies."""

    def __init__(self, in_channels):
        super(NonLocalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2  # 降维计算注意力，提高计算效率
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # 提取特征并调整维度
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C/2, H*W]
        g_x = g_x.permute(0, 2, 1)  # [B, H*W, C/2]

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, C/2, H*W]
        theta_x = theta_x.permute(0, 2, 1)  # [B, H*W, C/2]

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C/2, H*W]

        # 计算注意力图
        attention = torch.bmm(theta_x, phi_x)  # [B, H*W, H*W]
        attention = F.softmax(attention, dim=-1)

        # 计算输出特征
        y = torch.bmm(attention, g_x)  # [B, H*W, C/2]
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, H, W)  # [B, C/2, H, W]

        # 恢复通道数
        y = self.W(y)
        y = self.bn(y)

        return x + y  # 残差连接


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        """
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 通道压缩比例，默认32
        """
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]
        y = torch.cat([x_h, x_w], dim=2)  # 拼接后 [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # 将 y 按照 h 和 w 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_ca = x * a_h * a_w
        out_sa = self.self_SA_Enhance(out_ca)
        out = x * out_sa
        out = self.conv_end(out)
        return out

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div=2,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat',
                 upsample=True):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        self.upsample_flag = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        if self.up is not None:
            x = self.up(x)
        return x

class AFEM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AFEM, self).__init__()
        self.conv_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pcm = MLPBlock(dim=out_channels, upsample=False)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_adjust(x)
        x = self.pcm(x)
        return x


class Prompt_Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Prompt_Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


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


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class SFAP(nn.Module):
    def __init__(self):
        super(SFAP, self).__init__()
        self.var_enh = MyVarFeatureEnhance(
            input_dims=(144, 288, 576, 1152),
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
        )
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.mam1 = CoordAtt(64, 64)
        self.mam2 = CoordAtt(64, 64)
        self.mam3 = CoordAtt(64, 64)
        self.non_local4 = NonLocalAttention(64)

    def forward(self, x1, x2, x3, x4):
        x1, x2, x3, x4 = self.var_enh.forward_4(x1, x2, x3, x4)
        
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)
        
        x1 = self.mam1(x1)
        x2 = self.mam2(x2)
        x3 = self.mam3(x3)
        x4 = self.non_local4(x4)
        
        return x1, x2, x3, x4


class SDDNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SDDNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        # 删除不必要的模块
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        self.sfap = SFAP()
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Prompt_Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

        # 解码器部分采用 AFEM 模块
        self.up1 = AFEM(128, 64)
        self.up2 = AFEM(128, 64)
        self.up3 = AFEM(128, 64)
        self.up4 = AFEM(128, 64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: [B,4,H,W] -> 通过 conv4to3 变为 [B,3,H,W]
        # x = self.conv4to3(x)
        # encoder 提取特征
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.sfap(x1, x2, x3, x4)

        # decoder 部分
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2

if __name__ == "__main__":
    with torch.no_grad():
        model = SDDNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)
