import torch
import copy
from thop import profile

from network.starnet import starnet_s4

if __name__ == "__main__":
    device = "cuda"
    backbone = starnet_s4(True).to(device)
    test = backbone(torch.ones([1, 3, 128, 128]).float().to(device))

    for t in test:
        print(t.shape)

    n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input = torch.randn(1, 3, 128, 128).float().to(device)  # 输入尺寸需匹配模型

    # 计算FLOPs和参数量
    flops, _ = profile(copy.deepcopy(backbone), inputs=(test_input,))
    gflops = flops / 1e9  # 转换为GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f}")