#  HCGMNET: A HIERARCHICAL CHANGE GUIDING MAP NETWORK FOR CHANGE DETECTION,
#  IGARSS 2023,Oral. Chengxi. Han, Chen WU, Do Du,https://arxiv.org/abs/2302.10420
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchvision import models
import matplotlib.pyplot as plt

from network.MRA_com import MRA, DepthwiseSeparableConv, EMRA_focus
from network.RetinalChangeGuideModule import RetinalChangeGuideModule
from network.mutildattn import Mutildir_shaped_attention
from network.STRModule import STRModule
from network.starnet import starnet_s4, starnet_s2


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
        x = self.relu(x)
        return x



class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class merge_block(nn.Module):
    def __init__(self, inchannels,outchannels):
        super(merge_block, self).__init__()
        self.conv1 = DepthwiseSeparableConv(2*inchannels, outchannels, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv(2*inchannels, outchannels, 3, 1, 1)

    def forward(self,A,B,C):
        F1 = torch.cat([A,B],dim=1)
        F2 = torch.cat([A,C],dim=1)
        F3 = self.conv1(F1)
        F4 = self.conv2(F2)
        out = F.sigmoid(F3) * F4
        return out


class LwSRNet(nn.Module):
    def __init__(self,):
        super(LwSRNet, self).__init__()
        self.backbone = starnet_s2(pretrained=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.mdattn1 = EMRA_focus(32,7,nn.BatchNorm2d)
        self.mdattn2 = EMRA_focus(64,7,nn.BatchNorm2d)
        self.mdattn3 = EMRA_focus(128,7,nn.BatchNorm2d)
        self.mdattn4 = EMRA_focus(256,7,nn.BatchNorm2d)
        #
        self.strm1 = STRModule(32)
        self.strm2 = STRModule(64)
        self.strm3 = STRModule(128)
        self.strm4 = STRModule(256)

        self.conv_reduce_1 = DepthwiseSeparableConv(32*2,32,3,1,1)
        self.conv_reduce_2 = DepthwiseSeparableConv(64*2,64,3,1,1)
        self.conv_reduce_3 = DepthwiseSeparableConv(128*2,128,3,1,1)
        self.conv_reduce_4 = DepthwiseSeparableConv(256*2,256,3,1,1)

        # self.conv_reduce_5 = DepthwiseSeparableConv(32,32,3,1,1)
        # self.conv_reduce_6 = DepthwiseSeparableConv(64,64,3,1,1)
        # self.conv_reduce_7 = DepthwiseSeparableConv(128,128,3,1,1)
        # self.conv_reduce_8 = DepthwiseSeparableConv(256,256,3,1,1)


        self.conv_reduce_5 = merge_block(32,32)
        self.conv_reduce_6 = merge_block(64,64)
        self.conv_reduce_7 = merge_block(128,128)
        self.conv_reduce_8 = merge_block(256,256)

        # self.decoder = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde1 = nn.Sequential(BasicConv2d(32, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.deocde2 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.deocde3 = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.deocde4 = nn.Sequential(BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.deocde = nn.Sequential(BasicConv2d(480, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

        # self.decoder_final = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde_final = nn.Sequential(BasicConv2d(480, 64, 3, 1, 1),  nn.Conv2d(64, 1, 3, 1, 1))
        self.cgm_2 = RetinalChangeGuideModule(64,5)
        self.cgm_3 = RetinalChangeGuideModule(128,5)
        self.cgm_4 = RetinalChangeGuideModule(256,5)
        # self.cgm_2 = ChangeGuideModule(64)
        # self.cgm_3 = ChangeGuideModule(128)
        # self.cgm_4 = ChangeGuideModule(256)


    def batch_slice_A(self, A: torch.Tensor, batch: int) -> torch.Tensor:
        sliced = A[:batch]
        return sliced

    def batch_slice_B(self, A: torch.Tensor, batch: int) -> torch.Tensor:
        sliced = A[batch:]
        return sliced

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):
        size = A.size()[2:]
        batch = A.shape[0]
        all = torch.cat([A,B],dim=0)
        layer1, layer2, layer3, layer4 = self.backbone(all)

        layer1 = self.upsample(layer1)
        layer2 = self.upsample(layer2)
        layer3 = self.upsample(layer3)
        layer4 = self.upsample(layer4)

        layer1_A,layer1_B = self.batch_slice_A(layer1,batch),self.batch_slice_B(layer1,batch)
        layer2_A,layer2_B = self.batch_slice_A(layer2,batch),self.batch_slice_B(layer2,batch)
        layer3_A,layer3_B = self.batch_slice_A(layer3,batch),self.batch_slice_B(layer3,batch)
        layer4_A,layer4_B = self.batch_slice_A(layer4,batch),self.batch_slice_B(layer4,batch)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1_time = torch.cat((layer1_A.unsqueeze(1),layer1_B.unsqueeze(1)),dim=1)

        layer2_time = torch.cat((layer2_A.unsqueeze(1),layer2_B.unsqueeze(1)),dim=1)

        layer3_time = torch.cat((layer3_A.unsqueeze(1),layer3_B.unsqueeze(1)),dim=1)

        layer4_time = torch.cat((layer4_A.unsqueeze(1),layer4_B.unsqueeze(1)),dim=1)

        layer1_rd = torch.abs(layer1_B - layer1_A)
        res_map1 = self.mdattn1(layer1_rd)

        layer2_rd = torch.abs(layer2_B - layer2_A)
        res_map2 = self.mdattn2(layer2_rd)

        layer3_rd = torch.abs(layer3_B - layer3_A)
        res_map3 = self.mdattn3(layer3_rd)

        layer4_rd = torch.abs(layer4_B - layer4_A)
        res_map4 = self.mdattn4(layer4_rd)

        layer1_time = self.strm1(layer1_time)
        layer2_time = self.strm2(layer2_time)
        layer3_time = self.strm3(layer3_time)
        layer4_time = self.strm4(layer4_time)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        # layer1 = torch.cat((layer1,res_map1,layer1_time),dim=1)
        # layer2 = torch.cat((layer2,res_map2,layer2_time),dim=1)
        # layer3 = torch.cat((layer3,res_map3,layer3_time),dim=1)
        # layer4 = torch.cat((layer4,res_map4,layer4_time),dim=1)

        layer1 = self.conv_reduce_5(layer1,res_map1,layer1_time)
        layer2 = self.conv_reduce_6(layer2,res_map2,layer2_time)
        layer3 = self.conv_reduce_7(layer3,res_map3,layer3_time)
        layer4 = self.conv_reduce_8(layer4,res_map4,layer4_time)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        outputs = []
        outputs.append(F.interpolate(self.deocde1(layer1), size, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(self.deocde2(layer2), size, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(self.deocde3(layer3), size, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(self.deocde4(layer4), size, mode='bilinear', align_corners=True))

        feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)
        change_map = self.deocde(feature_fuse)
        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1)
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v2/' + 'v2-fuse-' + str(num) + '.png')
        # change_map = self.decoder(torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1))
        # ---------------注释这两句------------------------
        #change_map = change_map + F.interpolate(torch.abs(A-B).mean(dim=1,keepdim=True), layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)

        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        new_feature_fuse = torch.cat((layer1,layer2_2,layer3_2,layer4_2),dim=1)

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        outputs.append(change_map)
        final_map = self.deocde_final(new_feature_fuse)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return outputs, final_map


if __name__=='__main__':
    device = "cuda"
    model = LwSRNet().to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input1 = torch.randn(1, 3, 256, 256).float().cuda()  # 输入尺寸需匹配模型
    test_input2 = torch.randn(1, 3, 256, 256).float().cuda()    # 输入尺寸需匹配模型

    # 计算FLOPs和参数量
    flops, _ = profile(copy.deepcopy(model), inputs=(test_input1,test_input2))
    gflops = flops / 1e9  # 转换为GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f}")