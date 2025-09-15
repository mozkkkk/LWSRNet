import torch
import torch.nn as nn
import torch.nn.functional as F




def directional_sampling(input, direction, L, padding_mode='zeros'):
    """
    input: 输入特征图 [B, C, H, W]
    direction: 采样方向 ('horizontal', 'vertical', 'diagonal_left', 'diagonal_right')
    L: 采样长度 (奇数)
    padding_mode: 越界填充方式 ('zeros', 'border')
    返回: 采样结果 [(H*W), B, C, L]
    """
    B, C, H, W = input.shape
    d = (L - 1) // 2  # 半长

    # 生成基础坐标网格
    center_i, center_j = torch.meshgrid(torch.arange(H, device=input.device),
                                        torch.arange(W, device=input.device),
                                        indexing='ij')  # HxW

    # 根据方向生成偏移量
    if direction == 'horizontal':
        di = torch.zeros(L, device=input.device)
        dj = torch.arange(-d, d + 1, device=input.device)
    elif direction == 'vertical':
        di = torch.arange(-d, d + 1, device=input.device)
        dj = torch.zeros(L, device=input.device)
    elif direction == 'diagonal_left':
        di = torch.arange(-d, d + 1, device=input.device)
        dj = torch.arange(-d, d + 1, device=input.device)
    elif direction == 'diagonal_right':
        di = torch.arange(-d, d + 1, device=input.device)
        dj = torch.arange(d, -d - 1, step=-1, device=input.device)
    else:
        raise ValueError("方向需为 'horizontal', 'vertical', 'diagonal_left' 或 'diagonal_right'")

    # 生成采样坐标
    sampled_i = center_i.unsqueeze(-1) + di.view(1, 1, -1)  # HxWxL
    sampled_j = center_j.unsqueeze(-1) + dj.view(1, 1, -1)  # HxWxL

    # 处理越界坐标
    valid_i = (sampled_i >= 0) & (sampled_i < H)
    valid_j = (sampled_j >= 0) & (sampled_j < W)
    valid_mask = valid_i & valid_j  # HxWxL

    # 截断坐标到边界
    sampled_i = torch.clamp(sampled_i, 0, H - 1)
    sampled_j = torch.clamp(sampled_j, 0, W - 1)

    # 转换为整数索引
    sampled_i = sampled_i.long()
    sampled_j = sampled_j.long()

    # 从输入中索引值 [B, C, H, W] -> [B, C, H, W, L]
    output = input[:, :, sampled_i, sampled_j]

    # 处理填充模式
    if padding_mode == 'zeros':
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W, L)
        output = output * valid_mask

    # 调整形状为 (H*W)xBxCxL
    output = output.permute(2, 3, 0, 1, 4).contiguous().view(H * W, B, C, L)

    return output


class Mutildir_shaped_attention(nn.Module):
    def __init__(self, Channels, ratio=16, L=7):
        super(Mutildir_shaped_attention, self).__init__()
        self.sq_mod = nn.Conv2d(Channels, Channels // ratio, kernel_size=1, padding=0)
        self.extractor1 = nn.Linear(Channels // ratio * L, Channels // ratio)
        # self.extractor2 = nn.Linear(Channels//ratio*L, Channels//ratio)
        self.exp_mod = nn.Conv2d(Channels // ratio, Channels, kernel_size=1, padding=0)
        self.L = L

    def forward(self, inputs):
        inputs_q = self.sq_mod(inputs)
        b, c, h, w = inputs_q.shape
        output1 = directional_sampling(inputs_q, 'diagonal_right', self.L, padding_mode='zeros').permute(1, 0, 2,
                                                                                                         3).reshape(b,
                                                                                                                    h * w,
                                                                                                                    c * self.L)
        output2 = directional_sampling(inputs_q, 'diagonal_left', self.L, padding_mode='zeros').permute(1, 0, 2,
                                                                                                        3).reshape(b,
                                                                                                                   h * w,
                                                                                                                   c * self.L)
        output3 = directional_sampling(inputs_q, 'horizontal', self.L, padding_mode='zeros').permute(1, 0, 2,
                                                                                                     3).reshape(b,
                                                                                                                h * w,
                                                                                                                c * self.L)
        output4 = directional_sampling(inputs_q, 'vertical', self.L, padding_mode='zeros').permute(1, 0, 2, 3).reshape(
            b, h * w, c * self.L)
        output1 = self.extractor1(output1)
        output2 = self.extractor1(output2)
        output3 = self.extractor1(output3)
        output4 = self.extractor1(output4)
        output = (output1 + output2 + output3 + output4).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # output = (output1 + output2).reshape(b,h,w,c).permute(0,3,1,2)
        output = self.exp_mod(output)
        weights = F.sigmoid(output)  # b,h*w,c
        output = weights * inputs
        return output


if __name__ == "__main__":
    device = "cuda"

'''

# 创建输入张量 (B=1, C=1, H=3, W=3)
input1 = torch.tensor([[[
    [1, 2, 3, 4],
    [4, 5, 6, 5],
    [7, 8, 9, 6]
]]], dtype=torch.float32)
input1.requires_grad = True
print(input1.shape)
print(input1)
# 水平方向采样 L=3
output = directional_sampling(input1, 'diagonal_left', 3, padding_mode='zeros')
print(output.shape)  # torch.Size([9, 1, 1, 3])
print(output)  # torch.Size([9, 1, 1, 3])


output = output.reshape(4,3,1,1,3).permute(2,3,0,1,4)
print(output.shape)
print(output)
'''