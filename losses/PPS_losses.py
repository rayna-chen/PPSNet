# Our proposed loss functions involving Per-Pixel Shading (PPS) maps derived from an analytical Per-Pixel Lighting (PPL) calculation

import torch
import torch.nn.functional as F
import utils.optical_flow_funs as OF
from losses.calculate_PPL import calculate_per_pixel_lighting
def rgb_to_grayscale(input_tensor):
    # Assuming input_tensor is of shape (B, C, H, W) and C=3 for RGB
    return torch.matmul(input_tensor.permute(0, 2, 3, 1), torch.tensor([0.2989, 0.5870, 0.1140], device=input_tensor.device)).unsqueeze(1)

def normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero

def _image_derivatives(image, diff_type='center'):
    c = image.size(1)
    if diff_type == 'center':
        sobel_x = 0.5 * torch.tensor([[0.0, 0, 0], [-1, 0, 1], [0, 0, 0]], device=image.device).view(1, 1, 3, 3).repeat(
            c, 1, 1, 1)
        sobel_y = 0.5 * torch.tensor([[0.0, 1, 0], [0, 0, 0], [0, -1, 0]], device=image.device).view(1, 1, 3, 3).repeat(
            c, 1, 1, 1)
    elif diff_type == 'forward':
        sobel_x = torch.tensor([[0.0, 0, 0], [0, -1, 1], [0, 0, 0]], device=image.device).view(1, 1, 3, 3).repeat(c, 1,
                                                                                                                  1, 1)
        sobel_y = torch.tensor([[0.0, 1, 0], [0, -1, 0], [0, 0, 0]], device=image.device).view(1, 1, 3, 3).repeat(c, 1,
                                                                                                                  1, 1)

    dp_du = torch.nn.functional.conv2d(image, sobel_x, padding=1, groups=3)
    dp_dv = torch.nn.functional.conv2d(image, sobel_y, padding=1, groups=3)
    return dp_du, dp_dv


def _point_cloud_to_normals(pc, diff_type='center'):
    # pc (b,3,m,n)
    # return (b,3,m,n)
    dp_du, dp_dv = _image_derivatives(pc, diff_type=diff_type)
    normal = torch.nn.functional.normalize(torch.cross(dp_du, dp_dv, dim=1))
    return normal


def _get_normals_from_depth(depth, intrinsics, depth_is_along_ray=False, diff_type='center',
                            normalized_intrinsics=True):
    # depth (b,1,m,n)
    # intrinsics (b,3,3)
    # return (b,3,m,n), (b,3,m,n)
    dirs = OF.get_camera_pixel_directions(depth.shape[2:4], intrinsics, normalized_intrinsics=normalized_intrinsics)
    dirs = dirs.permute(0, 3, 1, 2)
    if depth_is_along_ray:
        dirs = torch.nn.functional.normalize(dirs, dim=1)
    pc = dirs * depth

    normal = _point_cloud_to_normals(pc, diff_type=diff_type)
    return normal, pc


def calculate_pps_corr_loss(img_tensor, l, a, normal):
    # Ensure L and A are in the same format as RGB (B, C, H, W)
    l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
    a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]

    # Log transformation on A
    a = torch.log(a + 1e-8)

    # Min-Max normalization for A should ideally be based on dataset statistics,
    # Here we normalize based on the min and max of the current batch for simplicity
    a_min, a_max = a.min(), a.max()
    a = (a - a_min) / (a_max - a_min + 1e-8)

    # Convert image to grayscale
    img_gray = rgb_to_grayscale(img_tensor)  # Resulting shape: (B, 1, H, W)
    # Define the threshold for specular highlights
    threshold = 0.98  # Adjust this threshold based on your data
    # Create a mask for pixels above the intensity threshold
    specular_mask = (img_gray > threshold).float()  # [B, 1, H, W]

    # Normalize l and normal
    l_norm = normalize_vectors(l)
    normal_norm = normalize_vectors(normal)


    # Compute dot product and apply attenuation
    dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    dot_product_clamped = torch.clamp(dot_product, -1, 1)
    dot_product_attenuated = dot_product_clamped * a

    # Apply the specularity mask by setting specular regions to zero in the spatial dimensions
    img_masked = img_gray * (1 - specular_mask)  # Apply mask directly on the grayscale image tensor
    dot_product_attenuated_masked = dot_product_attenuated * (1 - specular_mask)  # Apply mask to the attenuated dot product

    # Flatten the masked image tensor and masked dot product tensor for correlation computation
    img_flat_masked = img_masked.flatten(start_dim=2)  # Flatten spatial dimensions of the masked image
    dot_product_attenuated_flat_masked = dot_product_attenuated_masked.flatten(start_dim=2)  # Flatten spatial dimensions of the masked dot product

    # Recalculate correlation with masked values
    img_mean_subtracted_masked = img_flat_masked - img_flat_masked.mean(dim=2, keepdim=True)
    dot_product_mean_subtracted_masked = dot_product_attenuated_flat_masked - dot_product_attenuated_flat_masked.mean(dim=2, keepdim=True)

    cov_masked = (img_mean_subtracted_masked * dot_product_mean_subtracted_masked).sum(dim=2)
    img_std_masked = img_mean_subtracted_masked.pow(2).sum(dim=2).sqrt()
    dot_product_std_masked = dot_product_mean_subtracted_masked.pow(2).sum(dim=2).sqrt()

    correlation_coefficient_masked = cov_masked / (img_std_masked * dot_product_std_masked + 1e-8)

    # Calculate loss with masked correlation
    loss = 1 - correlation_coefficient_masked.mean()

    return loss

def calculate_pps_supp_loss(l_pred, a_pred, normal_pred, l_gt, a_gt, normal_gt):
    # Ensure L and A are in the same format as RGB (B, C, H, W)
    l_gt = l_gt.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
    a_gt = a_gt.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]
    l_pred = l_pred.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W]
    a_pred = a_pred.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W]

    # Log transformation on A
    a_gt = torch.log(a_gt + 1e-8)
    a_pred = torch.log(a_pred + 1e-8)

    # Min-Max normalization for A should ideally be based on dataset statistics,
    # Here we normalize based on the min and max of the current batch for simplicity
    a_gt_min, a_gt_max = a_gt.min(), a_gt.max()
    a_gt = (a_gt - a_gt_min) / (a_gt_max - a_gt_min + 1e-8)
    a_pred_min, a_pred_max = a_pred.min(), a_pred.max()
    a_pred = (a_pred - a_pred_min) / (a_pred_max - a_pred_min + 1e-8)

    # Normalize l and normal
    l_norm_gt = normalize_vectors(l_gt)
    normal_norm_gt = normalize_vectors(normal_gt)
    l_norm_pred = normalize_vectors(l_pred)
    normal_norm_pred = normalize_vectors(normal_pred)

    # Compute dot product and apply attenuation
    dot_product_gt = torch.sum(l_norm_gt * normal_norm_gt, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    dot_product_clamped_gt = torch.clamp(dot_product_gt, -1, 1)
    dot_product_attenuated_gt = dot_product_clamped_gt * a_gt

    dot_product_pred = torch.sum(l_norm_pred * normal_norm_pred, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    dot_product_clamped_pred = torch.clamp(dot_product_pred, -1, 1)
    dot_product_attenuated_pred = dot_product_clamped_pred * a_pred

    loss = F.mse_loss(dot_product_attenuated_pred, dot_product_attenuated_gt)
    
    return loss

def pps_supp_loss( pred_depth, gt_depth, ref_dirs, light_pos, light_dir, mu ,n_intrinsics):
    pc_pred = pred_depth.squeeze(1).unsqueeze(3) * ref_dirs
    pc_gt = gt_depth.squeeze(1).unsqueeze(3) * ref_dirs
    l_pred, a_pred = calculate_per_pixel_lighting(pc_pred, light_pos, light_dir, mu)
    l_gt, a_gt = calculate_per_pixel_lighting(pc_gt, light_pos, light_dir, mu)
    normal_pred,_ = _get_normals_from_depth(pred_depth, n_intrinsics, depth_is_along_ray=False, diff_type='center', normalized_intrinsics=True)
    normal_gt,_ = _get_normals_from_depth(gt_depth, n_intrinsics, depth_is_along_ray=False, diff_type='center', normalized_intrinsics=True)
    loss = calculate_pps_supp_loss(l_pred, a_pred, normal_pred, l_gt, a_gt, normal_gt)
    return loss


# if __name__ == '__main__':
#     from dataloaders.unity_dataloader import C3VD_Dataset
#     import argparse
#     from torchvision import transforms
#     import numpy as np
#     import PIL
#     from tqdm import tqdm
#
#     parser = argparse.ArgumentParser(description='Evaluate PPSNet')
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--data_dir', type=str, default='F:/monocular_depth_estimation_dataset/unity_endo')
#     parser.add_argument("--train_list", type=str, default="F:/monocular_depth_estimation_dataset/unity_endo/train.txt")
#     args = parser.parse_args()
#
#     testSet = C3VD_Dataset(data_dir=args.data_dir, list=args.train_list, mode='Train')
#     testLoader = torch.utils.data.DataLoader(testSet, batch_size=4,
#                                              shuffle=False, num_workers=1, generator=torch.Generator())
#
#     image_size = 518
#     preproc_trans = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
#                                         transforms.CenterCrop(image_size),
#                                         ])
#
#
#
#
#
#     for batch in tqdm(testLoader):
#
#         pred_depth = np.ones((4, 1, 518, 518))
#         gt_depth = np.ones((4, 1, 518, 518))
#         gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
#         pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
#         gt_depth = preproc_trans(gt_depth).float()
#         pred_depth = preproc_trans(pred_depth).float()
#
#         ref_dirs = OF.get_camera_pixel_directions(gt_depth.shape[2:4], batch['n_intrinsics'],
#                                                   normalized_intrinsics=True).cuda()
#
#         light_data = batch['light_data']
#         light_pos, light_dir, mu = [d.cuda() for d in light_data]
#         n_intrinsics = batch['n_intrinsics'].cuda()
#         loss = pps_supp_loss(pred_depth, gt_depth, ref_dirs, light_pos, light_dir, mu, n_intrinsics)
#         print(loss)

