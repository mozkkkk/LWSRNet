import torch
import torch.nn as nn
import torch.nn.functional as F

class STRModule(nn.Module):
    def __init__(self, in_channels):
        super(STRModule, self).__init__()

        self.fc = nn.Linear(8*8*2,8*8)

        self.norm = nn.BatchNorm2d(in_channels)

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=in_channels,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )


    def forward(self, x):
        # x 的形状为 (batch_size, time_steps, channels, height, width)
        batch_size, time_steps, channels, height, width = x.shape
        x1 = x.reshape(batch_size, time_steps, channels, height//8, 8, width//8, 8)
        x1 = x1.permute(0,2,3,5,1,4,6)
        x1 = x1.reshape(batch_size,channels*height//8*width//8,time_steps*8*8)

        # 重塑输入数据
        x = x.permute(0, 3, 4, 1, 2)  # 调整维度顺序
        x2 = x.reshape(batch_size * height * width, time_steps, channels)  # 合并 batch 和空间维度

        x1 = self.fc(x1).reshape(batch_size,channels,height//8,width//8,8,8).permute(0,1,2,4,3,5).reshape(batch_size,channels,height,width)
        x1 = self.norm(x1)
        x2,_ = self.lstm(x2)
        x2 = x2[:,-1,:]
        out_c = x2.shape[1]
        x2 = x2.reshape(batch_size, height, width, out_c).permute(0, 3, 1, 2)
        #x = F.sigmoid(x1) * x2
        x = x1+x2

        return x


# 示例用法
if __name__ == "__main__":
    # 假设输入形状为 (batch_size, time_steps, channels, height, width)
    batch_size = 2
    time_steps = 2
    channels = 64
    height = 16
    width = 16
    in_channels = channels
    out_channels = 64
    kernel_size = 3

    # 创建随机输入
    x = torch.randn(batch_size, time_steps, channels, height, width)

    # 创建模块
    module = STRModule(in_channels)

    # 前向传播
    output = module(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)