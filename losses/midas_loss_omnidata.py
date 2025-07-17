# Based on https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

import torch
import torch.nn as nn
import numpy as np

from losses.masked_losses_omnidata import masked_l1_loss


"""
https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/losses/midas_loss.py
"""
cuda0 = torch.device('cuda:0')

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = np.nan
    depth_gt_nan[~mask_valid] = np.nan

    mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)


    t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    # if divisor == 0:
    #     return 0
    # else:
    #     return torch.sum(image_loss) / divisor
    if divisor <= 1e-6:  # 避免除零错误
        return torch.tensor(0.0, device=image_loss.device)
    return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    # valid = M.nonzero()
    # image_loss[valid] = image_loss[valid] / M[valid]

    valid_mask = (M > 1e-6)
    if not valid_mask.any():
        return torch.tensor(0.0, device=image_loss.device)

    image_loss[valid_mask] = image_loss[valid_mask] / M[valid_mask]

    return torch.mean(image_loss)



def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2)).clamp(min=1.0)  # 确保最小值1

    diff = prediction - target
    diff = torch.mul(mask, diff)

    # grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    # mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    # grad_x = torch.mul(mask_x, grad_x)
    #
    # grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    # mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    # grad_y = torch.mul(mask_y, grad_y)
    #
    #
    # image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    #
    # return reduction(image_loss, M)
    # 计算x方向梯度
    grad_x = diff[:, :, 1:] - diff[:, :, :-1]
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])

    # 计算y方向梯度
    grad_y = diff[:, 1:, :] - diff[:, :-1, :]
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])

    # 使用Huber损失替代绝对值 (更鲁棒)
    huber_x = torch.where(torch.abs(grad_x) < 1.0,
                          0.5 * grad_x ** 2,
                          torch.abs(grad_x) - 0.5)
    huber_y = torch.where(torch.abs(grad_y) < 1.0,
                          0.5 * grad_y ** 2,
                          torch.abs(grad_y) - 0.5)

    # 应用掩码
    huber_x = torch.mul(mask_x, huber_x)
    huber_y = torch.mul(mask_y, huber_y)

    image_loss = torch.sum(huber_x, (1, 2)) + torch.sum(huber_y, (1, 2))

    return reduction(image_loss, M)



class SSIMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_preds, depth_gt, mask_valid):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid)
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid)
        return ssi_mae_loss


class GradientMatchingTerm(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

# orginal midas loss
# class MidasLoss(nn.Module):
#     def __init__(self, alpha=0.1, scales=4, reduction='image-based'):
#         super().__init__()
#
#         self.__ssi_mae_loss = SSIMAE()
#         self.__gradient_matching_term = GradientMatchingTerm(scales=scales, reduction=reduction)
#         self.__alpha = alpha
#         self.__prediction_ssi = None
#
#     def forward(self, prediction, target, mask):
#         prediction_inverse = 1 / (prediction.squeeze(1)+1e-6)
#         target_inverse = 1 / (target.squeeze(1)+1e-6)
#         ssi_loss = self.__ssi_mae_loss(prediction, target, mask)
#
#         scale, shift = compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))
#         self.__prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
#         reg_loss = self.__gradient_matching_term(self.__prediction_ssi, target_inverse, mask.squeeze(1))
#         if self.__alpha > 0:
#             total = ssi_loss + self.__alpha * reg_loss
#
#         return total, ssi_loss, reg_loss


class MidasLoss(nn.Module):
    def __init__(self, scales=4, reduction='image-based'):
        super().__init__()

        self.__ssi_mae_loss = SSIMAE()
        self.__gradient_matching_term = GradientMatchingTerm(scales=scales, reduction=reduction)
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        # prediction_inverse = 1 / (prediction.squeeze(1)+1e-6)
        # target_inverse = 1 / (target.squeeze(1)+1e-6)
        # ssi_loss = self.__ssi_mae_loss(prediction, target, mask)
        #
        # scale, shift = compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))
        # self.__prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
        # reg_loss = self.__gradient_matching_term(self.__prediction_ssi, target_inverse, mask.squeeze(1))
        #
        #
        # return ssi_loss, reg_loss

        clamped_depth = prediction.squeeze(1).clamp(min=0.01, max=10.0)  # 限制深度范围
        prediction_inverse = 1 / (clamped_depth + 1e-6)

        clamped_target = target.squeeze(1).clamp(min=0.01, max=10.0)
        target_inverse = 1 / (clamped_target + 1e-6)

        ssi_loss = self.__ssi_mae_loss(prediction, target, mask)
        scale, shift = compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))

        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
        reg_loss = self.__gradient_matching_term(self.__prediction_ssi, target_inverse, mask.squeeze(1))

        ### 修改点4：动态加权 ###
        # 根据有效像素比例调整梯度损失权重
        valid_ratio = mask.sum() / mask.numel()
        alpha = 0.1 * valid_ratio.clamp(min=0.1, max=1.0)  # 有效像素少时降低权重

        return ssi_loss, alpha * reg_loss

# if __name__ == '__main__':
#     import cv2
#     midas_loss = MidasLoss()
#     pred_depth = np.ones((4, 1, 512, 512)) - 0.05
#     gt_depth = np.ones((4, 1, 512, 512)) - 0.05
#     mask = ((gt_depth >= 0.0) & (gt_depth <= 0.98)).astype(np.float32)
#     gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
#     pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
#     mask = torch.tensor(np.asarray(mask, np.bool_)).cuda()
#     ssi_loss, reg_loss = midas_loss(pred_depth, gt_depth, mask)
#     print(ssi_loss)
#     print(reg_loss)