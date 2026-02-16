# PyTorch lib
import argparse
import os

import cv2
# Tools lib
import numpy as np
import torch
from torch.autograd import Variable
# Models lib
# Metrics lib
# from metrics import calc_psnr, calc_ssim
# from models import *
import torch.nn as nn
from ultralytics.nn.modules.LeWinTransformerBlock import LeWinTransformerBlock
from DBITNet_train import Generator
# from QTNet_Oringal import Generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--Normal_feak", type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image):
    # 是否进行消极残差连接
    infer_flag = True

    image = np.array(image, dtype='float32') / 255.
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    mean = np.array(mean, dtype='float32')
    std = np.array(std, dtype='float32')
    image = image - mean
    image = image / std
    image = image[:, :, (2, 1, 0)]
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).to(device)
    # Cityscapes
    out = model(image, infer_flag, device)[5]
    # out = model(image, infer_flag, device)[4]
    # # RTTS
    # out = model(image,device)[4]


    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]
    out = out[:, :, (2, 1, 0)]
    out = out * std
    out = out + mean
    out = out * 255.
    return out


if __name__ == '__main__':
    args = get_args()
    if args.mode == "normal_to_adverse":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Normal_train/'  # normal image 2975
        # args.output_dir = 'F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_feak/'
        args.output_dir = 'F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Normal_feak2/'
        args.name_list = './Normal_feak.txt'
    elif args.mode == "adverse_to_normal":
        # args.input_dir = './dataset/Normal_to_Foggy/images/Foggy_train/'  #
        # # Foggy-Cityscapes
        # args.output_dir = 'F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Foggy_feak1/'
        args.output_dir = 'F:/Datasets/heavy_weather/Cityscapes_Nomal_Foggy/Foggy_feak3/'
        args.name_list = './Foggy_feak.txt'
        # RTTS
        # args.output_dir = 'F:/Datasets/heavy_weather/UnannotatedHazyImages_feak1/'
        args.name_list = './UnannotatedHazyImages_feak.txt'
    path_to_output_dir = os.path.join(args.output_dir)
    path_weight = os.path.join(args.weight)
    os.makedirs(path_to_output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator().to(device)
    model.load_state_dict(torch.load(path_weight))

    input_list = sorted(os.listdir(args.input_dir))
    num = len(input_list)
    with open(args.name_list, 'a') as tx:
        for i in range(num):
            print('Processing image: %s' % (input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            original_size = img.shape
            # img = align_to_four(img)
            dsize = (416, 416)
            img = cv2.resize(img, dsize)
            result = predict(img)
            size = (original_size[1], original_size[0])
            result = cv2.resize(result, size)
            # img_name = input_list[i].split('.')[0]
            img_name = input_list[i]
            tx.write('source_' + img_name[:-4])
            tx.write('\n')
            # cv2.imwrite(args.output_dir + 'source_' + img_name[:-4] + '_fake_B.png', result)
            # Foggy-Cityscapes
            cv2.imwrite(args.output_dir + 'source_' + img_name[:-4] + '_fake_B.png', result)
            # # RTTS
            # cv2.imwrite(args.output_dir + img_name[:-4] + '_fake.png', result)

        tx.close()

