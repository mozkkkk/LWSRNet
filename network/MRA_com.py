import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        # 深度卷积（不改变通道数）
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # 关键参数：groups=in_channels
        )
        # 逐点卷积（1x1卷积调整通道数）
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 可选：添加BatchNorm和激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FocusDownsample(nn.Module):
    """
    Reorganizes a tensor from (B, C, H, W) to (B, 9C, H/3, W/3).
    Pads the input if its spatial dimensions are not divisible by 3.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # Calculate padding needed for height and width
        # The (..., % 3) handles cases where the dimension is already a multiple of 3
        pad_h = (3 - H % 3) % 3
        pad_w = (3 - W % 3) % 3

        # Apply padding to the bottom and right
        # Format for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Slicing will now work correctly on the padded tensor
        return torch.cat([
            x[..., 0::3, 0::3],
            x[..., 0::3, 1::3],
            x[..., 0::3, 2::3],
            x[..., 1::3, 0::3],
            x[..., 1::3, 1::3],
            x[..., 1::3, 2::3],
            x[..., 2::3, 0::3],
            x[..., 2::3, 1::3],
            x[..., 2::3, 2::3],
        ], dim=1)


class EMRA_focus(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel

        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.focus_downsample = FocusDownsample()

        # 核心改变：引入一个统一的、高效的通道融合模块
        # 使用分组卷积，将 9*C 的通道分为 9 组，每组内部进行 1x1 卷积
        # 参数量为 9 * (C * C) = 9*C^2，远低于普通 1x1 卷积的 (9C)^2
        self.channel_mixer = nn.Conv2d(9 * channel, 9 * channel, kernel_size=1, groups=9, bias=False)
        self.mixer_norm = norm_layer(9 * channel)

        # 注意力计算部分保持不变，它本身是高效的
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

        self.focus_upsample = nn.PixelShuffle(upscale_factor=3)

    def forward(self, x):
        original_H, original_W = x.shape[-2:]
        x_tem_pool = self.max_m1(x)

        # 1. FocusDownsample 得到 9*C 特征
        x_focused = self.focus_downsample(x_tem_pool)

        # 2. 统一进行通道融合
        x_mixed = self.channel_mixer(x_focused)
        x_mixed = self.mixer_norm(x_mixed)

        # 3. 功能分离：从融合后的特征中拆分
        x_tem, x_key = x_mixed.split([self.channel, 8 * self.channel], dim=1)

        # 4. 对 x_tem 进行高效的轴向注意力计算
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        att_map = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        # 5. 将注意力图与已经处理过的 x_key 特征直接拼接
        # x_key 不再需要任何额外计算
        final_features = torch.cat([att_map, x_key], dim=1)

        # 6. 上采样并应用注意力
        att = self.focus_upsample(final_features)
        att = att[..., :original_H, :original_W]  # 裁剪以匹配原始尺寸

        out = x * self.gate_fn(att)  # 此处 gate_fn 应用在att上，而不是原始x上

        return out
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


if __name__ == "__main__":
    device = "cuda"
    rcg = EMRA_focus(256,7,nn.BatchNorm2d).to(device)
    n_parameters = sum(p.numel() for p in rcg.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input1 = torch.randn(4, 256, 64, 64).float().cuda()  # 输入尺寸需匹配模型
    output = rcg(test_input1)
    print(output.shape)