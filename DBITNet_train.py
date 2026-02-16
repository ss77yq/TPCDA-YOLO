# -*- coding: UTF-8 -*-
# PyTorch lib
import argparse
import math
import os
import time
import cv2
# Tools lib
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
from ultralytics.nn.modules.LeWinTransformerBlock import LeWinTransformerBlock
from ultralytics.utils.img_utils import *


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

# 上采样
class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

########## Loss ###########
# CharbonnierLoss：L1损失（改进版）
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


# 通道注意力
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

# 监督注意力
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

# ##################
# OBSNet网络
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# OBSNet
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats=16, kernel_size=3, reduction=4, bias=False,  num_cab=2):
        super(ORSNet, self).__init__()

        act = nn.PReLU()
        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_dec1 = UpSample(n_feat, 128)
        self.up_dec2 = nn.Sequential(UpSample(128, 256), UpSample(n_feat, 128))

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    # def forward(self, x, encoder_outs, decoder_outs):
    def forward(self, x, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


# Tools lib
# import cv2
# 生成器模型:属于某域的图像，生成另一个域的图像
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # conv1：输入通过一个从3通道到64通道的卷积层
        # conv2 和 conv3：进一步的卷积操作，将通道深度增加到128，同时减少空间维度（将图像尺寸减半至0.5）
        # conv4, conv5, conv6：继续卷积操作，将通道深度增加到256（将图像尺寸减半至0.25）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        # neck
        # diconv1, diconv2, diconv3, diconv4：四层膨胀卷积用于在不降低分辨率的情况下捕获更广泛的上下文信息
        # 其中膨胀系数分别为2, 4, 8, 16
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )

        # conv7 和 conv8：继续使用标准卷积进行处理
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        # deconv1：双线性从256通道上采样到128通道(将图像尺寸放大一倍(0.5))
        # conv9：上采样后的细化处理
        # deconv2：进一步双线性上采样到64通道(将图像尺寸再次放大一倍(1)
        # conv10：输出前的最后细化处理
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        # outframe1 和 outframe2：从不同阶段生成中间帧（生成两个不同尺度图像，方便后续进行多尺度Lr损失计算）
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )
        # output：生成最终输出图像
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

        act = nn.PReLU()
        # CAB注意力模块
        self.skip_attn1 = CAB(64, kernel_size=3 , reduction=4, bias=False, act=act)
        self.skip_attn2 = CAB(128, kernel_size=3, reduction=4, bias=False, act=act)

        # SAM注意力模块
        self.sam = SAM(64, kernel_size=1, bias=False)

        # ORSNet前序准备
        self.shallow = nn.Sequential(conv(3, 64, kernel_size=3, bias=False), CAB(64, kernel_size=3, reduction=4, bias=False, act=act))
        self.concat = conv(64 * 2, 64 + 32, kernel_size=3, bias=False)

        # ORSNet
        self.orsnet = ORSNet(64, scale_orsnetfeats=32, kernel_size=3, reduction=4, bias=False, num_cab=2)
        self.tail   = conv(64+32, 32, kernel_size=3, bias=False)



    def forward(self, input, infer_flag, device):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        mask_list = []
        attention_map = []
        feat2 = []

        # Conv1 (3 → 64)
        x = self.conv1(input)
        res1 = x

        # Conv2 (64 → 128) Conv3 (128 → 128)
        # 通过Conv2 步长为2 ，尺寸减半(0.5)
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x

        # Conv4 (128 → 256) Conv5 (256 → 256) Conv6 (256 → 256)
        # 通过Conv4 步长为2 ，尺寸减半(0.25)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)


        # neck部分
        # 膨胀卷积Diconv1 (d=2) Diconv2 (d=4) Diconv3 (d=8) Diconv4 (d=16)
        # 增加卷积核感受野，但不改变图像尺寸
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)



        #  Conv7 (256 → 256) Conv8 (256 → 256)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        feat2.append(x)

        # Deconv1 (256 → 128)
        # 反卷积操作将图像尺寸放大一倍(0.5)
        x = self.deconv1(x)
        if x.shape != res2.shape:
            print('!ok')
        x = x + self.skip_attn2(res2)
        # x = x + res2
        # Conv9 (128 → 128)
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        feat2.append(x)

        # Deconv2 (128 → 64)
        x = self.deconv2(x)
        # 通过反卷积操作将图像尺寸再次放大一倍(1)
        if x.shape != res1.shape:
            print('!ok')
        x = x + self.skip_attn1(res1)
        # x = x + res1
        feat2.append(x)

        # 修改
        # 双分支：ORSNet（不改变分辨率情况下进行实现）
        x3_samfeats, img_mid =self.sam(x, input)
        x3 = self.shallow(input)
        x3_cat = self.concat(torch.cat([x3, x3_samfeats], 1))
        feat2 = feat2[::-1]     # 此时feat2是反的，进行倒置
        x3 = self.orsnet(x3_cat, feat2)
        stage3_img = self.tail(x3)
        stage3_img = self.output(stage3_img)
        # 消极残差连接
        if infer_flag:
            print("OBSNet正在推断---")
            stage3_img = stage3_img + input     # 消极残差连接

        # Conv10 (64 → 32)
        x = self.conv10(x)
        # Output (32 → 3)
        x = self.output(x)
        # 消极残差连接
        # 只在推理阶段进行，训练阶段不进行残差连接
        # 训练时，保证图像质量和精度都按照图像重建正常进行（如果加入，可能会阻碍损失不下降）
        # 测试时，加入消极残差连接，使其介于两个域之间的内插图像
        if infer_flag:
            x = input + x
            print("正在推断---")
        return mask_list, frame1, frame2, attention_map, x, stage3_img


# 图像转化为tensor，并应用归一化
def prepare_img_to_tensor(image, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    image = np.array(image, dtype='float32') / 255.
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image - mean
    image = image / std
    image = image[:, :, (2, 1, 0)]
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = image.to(device)
    return image


# 设置模型参数是否需要梯度更新，即是否在训练中更新权重
def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable


# 初始化一个预训练的VGG16模型，用于提取特征，用于感知损失（perceptual loss）
# Initialize VGG16 with pretrained weight on ImageNet
def vgg_init(device, model_weights):
    vgg_model = torchvision.models.vgg16()
    vgg_model.load_state_dict(torch.load(model_weights))
    vgg_model.to(device)
    # vgg_model = vgg_model.classifier[:-1]
    vgg_model.eval()
    trainable(vgg_model, False)
    return vgg_model

def res_init(device, model_weights):
    res_model = torchvision.models.resnet18(pretrained=True)
    # res_model.load_state_dict(torch.load(model_weights))
    res_model.to(device)
    # vgg_model = vgg_model.classifier[:-1]
    res_model.eval()
    trainable(res_model, False)
    return res_model



# 从VGG16模型中提取中间层的特征，这些特征在感知损失perceptual loss计算中有用
# Extract features from internal layers for perceptual loss
class Vgg(nn.Module):
    def __init__(self, vgg_model):
        super(Vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        # 定义了字典：在VGG16中提取哪些层的输出
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        # output：包含了从选定VGG16层中提取的特征图
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output


class Res18(nn.Module):
    def __init__(self, resnet_model):
        super(Res18, self).__init__()
        # 提取ResNet的初始卷积层和各个残差块组
        self.resnet_layers = nn.Sequential(*list(resnet_model.children())[:8])

        # 定义字典：指定从哪些层提取输出
        self.layer_name_mapping = {
            '4': "conv1",  # 初始卷积层后的输出
            '5': "layer1",  # 第一个残差块组后的输出
            '6': "layer2",  # 第二个残差块组后的输出
            '7': "layer3"  # 第三个残差块组后的输出
            # 如果需要从更多层提取，可以继续添加
        }

    def forward(self, x):
        output = []
        for name, module in self.resnet_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output


# 生成器权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 调整图像尺寸（scale_coefficient为缩放因子）
def resize_image(image, scale_coefficient):
    # calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_coefficient)
    height = int(image.shape[0] * scale_coefficient)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(image, dsize)
    return output


# 损失函数计算
def loss_generator(generator_results, back_ground_truth):
    # 均采用均方损失
    mseloss = nn.MSELoss()

    # 返回生成器三种尺度的输出
    _s = [generator_results[1], generator_results[2], generator_results[4], generator_results[5]]
    # _s = [generator_results[1], generator_results[2], generator_results[4]]
    # 对标签进行尺寸调整和张量化
    _t = [prepare_img_to_tensor(resize_image(back_ground_truth, 0.25)),
          prepare_img_to_tensor(resize_image(back_ground_truth, 0.5)),
          prepare_img_to_tensor(back_ground_truth),
          prepare_img_to_tensor(back_ground_truth)
          ]
    # _t = [prepare_img_to_tensor(resize_image(back_ground_truth, 0.25)),
    #       prepare_img_to_tensor(resize_image(back_ground_truth, 0.5)),
    #       prepare_img_to_tensor(back_ground_truth)
    #       ]
    # 定义Lr损失中的权重参数
    _lamda = lamda_in_autoencoder

    # 计算Lr损失（多尺度重建损失）
    # Lr(S, T) = Σλi * MSE(Si, Ti)
    lm_s_t = 0
    for i in range(len(_s)):
        lm_s_t += _lamda[i] * mseloss(_s[i], _t[i])
    lm_s_t = torch.mean(lm_s_t)


    # 计算Lp损失（感知损失）
    # Lp(O, T) = MSE(VGG(O), VGG(T))
    lp_o_t = 0
    # loss2 = nn.MSELoss()
    # 只对生成的最后图像的单一尺寸进行感知损失计算
    # 通过VGG后，生成四个尺度的张量列表，分别进行MSE损失计算（最后进行取平均值）
    # 修改
    # vgg_to_gen = vgg16(generator_results[4])
    vgg_to_gen = vgg16(generator_results[5])
    vgg_to_gt = vgg16(prepare_img_to_tensor(back_ground_truth))
    for i in range(len(vgg_to_gen)):
        lp_o_t += mseloss(vgg_to_gen[i], vgg_to_gt[i])
    lp_o_t = torch.mean(lp_o_t)

    # # 修改
    # res_to_gen = res18(generator_results[4])
    # res_to_gt = res18(prepare_img_to_tensor(back_ground_truth))
    # for i in range(len(res_to_gen)):
    #     lp_o_t += mseloss(res_to_gen[i], res_to_gt[i])
    # lp_o_t = torch.mean(lp_o_t)

    # LGAN(O) = log(1 - D(G(I)))
    # 对两种损失进行相加
    l_g = (0.2 * lm_s_t) + (0.8 * lp_o_t)
    # l_g = lm_s_t + lp_o_t
    return l_g, lm_s_t, lp_o_t


# 参数设置
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal_to_adverse", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args


# 进行训练
def train():
    index = 0
    input_list = sorted(os.listdir(args.input_dir))
    # input_list = os.listdir(args.input_dir)
    gt_list = sorted(os.listdir(args.gt_dir))
    # gt_list = os.listdir(args.gt_dir)
    generator.apply(weights_init)
    best_psnr = 0
    best_epoch_p = 0
    best_ssim = 0
    best_epoch_s = 0

    # 训练
    infer_flag = False
    for _e in range(previous_epoch + 1, epoch):
        # 在训练开始前记录时间
        start_time = time.time()
        print("======finish  ", _e, ' / ', epoch, "==========")

        for _i in range(len(input_list)):
            img = cv2.imread(args.input_dir + input_list[_i])
            gt = cv2.imread(args.gt_dir + gt_list[_i])
            # 问题：图像质量不佳
            # 可以对这一步不进行重设置尺寸（会有点消耗计算资源）
            dsize = (416, 416)
            img = cv2.resize(img, dsize)
            gt = cv2.resize(gt, dsize)
            img_tensor = prepare_img_to_tensor(img)
            result = generator(img_tensor, infer_flag, device)
            loss1 = loss_generator(result, gt)

            optimizer_g.zero_grad()
            # Backpropagation
            loss1[0].backward()
            optimizer_g.step()

            torch.save(generator.state_dict(), os.path.join(args.save_weight,
                                                            '_' + str(_e) + '.pth')
                       )

        # 验证
        if _e > 0:
            generator.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            infer_flag = False
            for _j in range(5):
                img_begin = np.array(Image.open(args.input_dir + input_list[_j]))
                gt_begin = np.array(Image.open(args.gt_dir + gt_list[_j]))
                dsize = (416, 416)
                img_begin = cv2.resize(img_begin, dsize)
                gt_begin = cv2.resize(gt_begin, dsize)
                img = prepare_img_to_tensor(img_begin)
                gt = prepare_img_to_tensor(gt_begin)
                print("正在验证---")

                with torch.no_grad():
                    restored = generator(img, infer_flag, device)
                # restored = restored[4]
                # 修改
                restored = restored[5]

                for res, tar in zip(restored, gt):
                    psnr_val_rgb.append(torchPSNR(res, tar))

                    res = np.array(res.cpu().numpy())
                    res = np.transpose(res, (1, 2, 0))
                    tar = np.array(tar.cpu().numpy())
                    tar = np.transpose(tar, (1, 2, 0))
                    ssim_val_rgb.append(numpySSIM(res, tar))


            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()


            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_p = _e
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_s = _e

            print(
                "epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f" % (_e, psnr_val_rgb, best_epoch_p, best_psnr))
            print(
                "epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f" % (_e, ssim_val_rgb, best_epoch_s, best_ssim))


        # 在训练结束后记录时间
        end_time = time.time()
        # 计算并打印训练时间
        train_time = end_time - start_time

        print(f"epoch {_e}，总损失为：{loss1[0]},Lr损失为：{loss1[1]},Lp损失为：{loss1[2]}", "训练时间为: ", train_time,"s")






if __name__ == '__main__':
    args = get_args()
    if args.mode == "normal_to_adverse":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Normal_train/'  # normal image 2975
        # args.gt_dir = './dataset/Normal_to_Foggy/images/Foggy_train/' # adverse image 2975
        # args.save_weight = './QTNet_run/QTNet_weights/normal_to_foggy_fin_OBSNet/'
        args.save_weight = './DBITNet_run/DBITNet_weights/normal_to_foggy_Rain_fin11/'
    elif args.mode == "adverse_to_normal":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Foggy_train/'  #
        # args.gt_dir = './dataset/Normal_to_Foggy/images/Normal_train/' #
        args.save_weight = './DBITNet_run/DBITNet_weights/foggy_to_normal_Rain_fin11/'
    args.demo_img = './demo/output_foggy_drop_res/'
    path_to_save = os.path.join(args.save_weight)
    os.makedirs(path_to_save, exist_ok=True)

    model_weights = './QTNet_run/vgg16_caffe.pth'
    previous_epoch = 0
    epoch = 50
    learning_rate = 0.0002
    mean = (0.406, 0.456, 0.485)

    std = (0.225, 0.224, 0.229)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator().to(device)
    vgg16 = Vgg(vgg_init(device, model_weights))
    # # 修改
    # res18 = Res18(res_init(device, model_weights))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    # lamda_in_autoencoder = [0.01, 0.01, 0.01, 0.01]
    lamda_in_autoencoder = [0.6, 0.8, 1, 1]
    # lamda_in_autoencoder = [0.01, 0.01, 0.01]
    print("Start Training...")
    train()
    print("Training Finished...")