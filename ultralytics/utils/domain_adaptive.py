import torch
import torch.nn.functional as F

# # 对于binary_cross_entropy_with_logits的重写：添加了空间注意力
# def comput_loss(prob, y, att, size_average=True):
#         prob = torch.sigmoid(prob)
#         loss = -(y * torch.log(prob) + (1 - y) * torch.log(1 - prob))
#
#         # 如果提供了权重，将损失乘以权重
#         loss = loss * att
#
#         # 如果 size_average 是 True，则对平均 batch 大小进行归一化
#         if size_average:
#                 return loss.mean()
#         else:
#                 return loss.sum()


def prepare_masks(targets, is_source_mark):
        masks = []
        # targets = torch.tensor([1,2,3,4,5,6,7,8]).to("cuda:0")
        for targets_per_image in targets:
            is_source = targets_per_image
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source_mark else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

def loss_eval( domain_1, target, is_source_mark):
        masks = prepare_masks(target, is_source_mark)
        masks = torch.cat(masks, dim=0)
        N, A, H, W = domain_1.shape
        da_img_per_level = domain_1.permute(0, 2, 3, 1)
        da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
        masks = masks.bool()
        # print("masks: ",masks)
        da_img_label_per_level[masks, :] = 1

        da_img_per_level = da_img_per_level.reshape(N, -1)
        da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        da_img_labels_flattened = da_img_label_per_level
        da_img_flattened = da_img_per_level
        da_img_labels_flattened = da_img_labels_flattened
        # 二元交叉熵损失函数（多了个sigmod激活）
        da_img_loss1 = F.binary_cross_entropy_with_logits(da_img_flattened, da_img_labels_flattened)
        return da_img_loss1


# 在整张特征图的激活位置进行一致性正则化
def consistency_loss(img, pix):
    # 尺寸均为[8, 1, 80, 80]
    # 最大池化
    max_img, _ = torch.max(img, dim=1, keepdim=True)
    max_pix, _ = torch.max(pix, dim=1, keepdim=True)


    # 平均池化
    avg_img = torch.mean(img, dim=1, keepdim=True)
    avg_pix = torch.mean(pix, dim=1, keepdim=True)


    # 拼接形成([8, 2, 80, 80])
    F_img = torch.cat((max_img, avg_img), dim=1)
    F_pix = torch.cat((max_pix, avg_pix), dim=1)


    # DA_cst_loss = torch.nn.MSELoss()(F_pix, F_img)
    # 尝试下换一下位置
    DA_cst_loss = torch.nn.MSELoss()(F_img, F_pix)
    DA_cst_loss = DA_cst_loss.mean()
    return DA_cst_loss

# def BRM_loss(x_s, x_t):
#     # 源域，形成[8, 576, 1, 1]
#     max_x_s = F.adaptive_max_pool2d(x_s, (1, 1))
#     avg_x_s = F.adaptive_avg_pool2d(x_s, (1, 1))
#
#     # 目标域，形成[8, 576, 1, 1]
#     max_x_t = F.adaptive_max_pool2d(x_t, (1, 1))
#     avg_x_t = F.adaptive_avg_pool2d(x_t, (1, 1))
#
#     # 拼接形成[8, 1152, 1, 1]
#     F_s = torch.cat((max_x_s, avg_x_s), dim=1)
#     F_t = torch.cat((max_x_t, avg_x_t), dim=1)
#
#     # 计算MSE损失
#     DA_cst_loss = torch.nn.MSELoss()(F_s, F_t)
#     DA_cst_loss = DA_cst_loss.mean()
#     return DA_cst_loss
