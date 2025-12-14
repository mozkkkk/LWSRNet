import math

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import math


def normalize_grid_custom(grid, x_min, x_max, y_min, y_max):
    """
    Normalize grid coordinates from [x_min, x_max] and [y_min, y_max] to [-1, 1].
    """
    x = grid[..., 0]
    y = grid[..., 1]

    x_norm = 2*((x - x_min) / (x_max - x_min) - 0.5)
    y_norm = 2*((y - y_min) / (y_max - y_min) - 0.5)

    return torch.stack([x_norm, y_norm], dim=-1)

def circular_sampling(input, r, num_points=8, padding_mode='zeros'):
    """
    input: 输入特征图 [B, C, H, W]
    r: 半径
    num_points: 圆环上的采样点数
    padding_mode: 越界填充方式 ('zeros', 'border')
    返回: 采样结果 [H*W, B, C, num_points]
    """
    B, C, H, W = input.shape
    device = input.device

    # 生成基础坐标网格
    center_i, center_j = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij')  # [H, W]

    # 生成圆环上的角度 (均匀分布)
    angles = torch.linspace(0, 2 * math.pi, num_points + 1, device=device)[:-1]  # [num_points]
    di = r * torch.sin(angles)  # [num_points]
    dj = r * torch.cos(angles)  # [num_points]

    # 计算采样坐标 (浮点数)
    sampled_i = center_i.unsqueeze(-1) + di.view(1, 1, -1)  # [H, W, num_points]
    sampled_j = center_j.unsqueeze(-1) + dj.view(1, 1, -1)  # [H, W, num_points]

    if padding_mode == 'border':
        # 直接截断坐标
        sampled_i = torch.clamp(sampled_i, 0, H - 1)
        sampled_j = torch.clamp(sampled_j, 0, W - 1)

    sampled_i = sampled_i.repeat(B,1,1,1).unsqueeze(-1)
    sampled_j = sampled_j.repeat(B,1,1,1).unsqueeze(-1)
    grid = torch.cat([sampled_j,sampled_i],dim=-1)
    grid = normalize_grid_custom(grid,0,grid.shape[2]-1,0,grid.shape[1]-1)
    output = []
    for i in range(num_points):
        sub_grid = grid[..., i, :]  # (1, H, W, 2)
        out_i = F.grid_sample(input, sub_grid, mode='bilinear', padding_mode="zeros", align_corners=True)  # (1, C, H, W)
        output.append(out_i)
    output = torch.stack(output, dim=2)  # (B, C, 4, H, W)

    #output = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 调整形状为 (H*W)xBxCxnum_points
    output = output.permute(3, 4, 0, 1, 2).contiguous().view(H * W, B, C, num_points)

    return output


class RadialReceptiveField(nn.Module):
    """
    径向感受野模块
    """

    def __init__(self, in_channels, max_radius=5, num_points=8):
        super().__init__()
        self.in_channels = in_channels
        self.max_radius = max_radius
        self.num_points = num_points
        self.circle_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_points*in_channels, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, in_channels))
            for _ in range(max_radius)
        ])

    def forward(self,x):
        b, c, h, w = x.shape
        weights = []
        for r in range(self.max_radius):
            circular_feat = circular_sampling(x,r,self.num_points).permute(1, 0, 2, 3).reshape(b, h * w, c * self.num_points)
            circular_feat = self.circle_transform[r](circular_feat)
            weights.append(circular_feat)
        weights = F.sigmoid(torch.sum(torch.stack(weights),dim=0)).reshape(b, h, w, c).permute(0, 3, 1, 2)
        output = x*weights

        return output


class RetinalChangeGuideModule(nn.Module):
    """
    创新整合：
    1. LRSA局部精细处理
    2. 径向分区感受野模块
    """

    def __init__(self, in_dim, max_rad=11):
        super().__init__()
        self.chanel_in = in_dim

        # 外周通路：径向感受野处理
        self.peripheral_path = RadialReceptiveField(in_dim,max_rad)

        # 变化引导的注意力机制
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 动态调制参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.foveal_weight = nn.Parameter(torch.ones(1))
        self.peripheral_weight = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        # 调整变化图大小
        guiding_map = F.interpolate(guiding_map0, (height, width), mode='bilinear', align_corners=True)
        guiding_map = self.sigmoid(guiding_map)

        x_fused = self.peripheral_path(x)

        # ===== 变化引导注意力 =====
        # 使用变化图调制查询和键
        query = self.query_conv(x_fused) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x_fused) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x_fused) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x_fused

        return out

if __name__ == "__main__":
    device = "cuda"
    rcg = RadialReceptiveField(1,7,4).to(device)
    n_parameters = sum(p.numel() for p in rcg.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #test_input1 = torch.randn(1, 1, 3, 3).float().cuda()  # 输入尺寸需匹配模型
    test_input1 = torch.tensor([[1, 1], [3, 3]]).unsqueeze(0).unsqueeze(0).float().cuda()  # 输入尺寸需匹配模型
    print(test_input1.shape)
    print(test_input1)
    #test_input2 = torch.randn(4, 1, 64, 64).float().cuda()  # 输入尺寸需匹配模型

    output = rcg(test_input1)

    print(output.shape)
