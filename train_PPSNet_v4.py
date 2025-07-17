import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import os
from pathlib import Path
import numpy as np
import argparse
import logging
import pprint
from datetime import datetime
import utils.optical_flow_funs as OF
from dataloaders.unity_dataloader import C3VD_Dataset
from modules.PPSNet import PPSNet_Backbone, PPSNet_Refinement
from losses.combined_loss import CombinedLoss
from losses.eval_depth import eval_depth
from losses.depth_anything_v2_loss import SiLogLoss,GradL1Loss
from losses.midas_loss_omnidata import MidasLoss
from losses.VNL_loss_ominidata import VNL_Loss
from losses.PPS_losses import pps_supp_loss
from utils.ini_log import init_log
from torch.utils.tensorboard import SummaryWriter

# Set random seed for reproducibility
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)




def train_teacher(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)

    # Dataset and DataLoader
    dataset = C3VD_Dataset(data_dir=args.data_dir, list=args.train_list, mode='Train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=general_generator)
    testdata = C3VD_Dataset(data_dir=args.data_dir, list=args.test_list, mode='Train')
    testloader = DataLoader(testdata, batch_size=1, shuffle=True, num_workers=args.num_workers,
                            generator=general_generator)

    # Model
    model = PPSNet_Backbone.from_pretrained('LiheYoung/depth_anything_vits14')
    refinement_model = PPSNet_Refinement(1, 384)
    model.to(device)
    refinement_model.to(device)

    # Optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(refinement_model.parameters()), lr=args.lr,betas=(0.9, 0.999),weight_decay=0.01)
    lr = args.lr

    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    if args.pretrained_from:
        checkpoint = torch.load(args.pretrained_from, map_location=device)
        # checkpoint = torch.load(args.pretrained_from, map_location=map_location, weights_only=False)
        # backbone_state_dict = {}
        # for k, v in checkpoint['student_state_dict'].items():
        #     name = k[7:] if k.startswith('module.') else k
        #     backbone_state_dict[name] = v
        # model.load_state_dict(backbone_state_dict)
        #
        # refinement_state_dict = {}
        # for k, v in checkpoint['refiner_state_dict'].items():
        #     name = k[7:] if k.startswith('module.') else k
        #     refinement_state_dict[name] = v
        # refinement_model.load_state_dict(refinement_state_dict)


        model.load_state_dict(checkpoint['model'], strict=False)
        if 'refinement_model' in checkpoint:
            refinement_model.load_state_dict(checkpoint['refinement_model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])



    # Loss functions
    # criterion = CombinedLoss(scales=4, avg_fx=715, avg_fy=715, input_size=(518, 518)).to(device)
    # silog_criterion = SiLogLoss().to(device)
    # gradl1_criterion = GradL1Loss().to(device)
    midas_criterion = MidasLoss().to(device)
    vnl_criterion = VNL_Loss(9.5080e-01, 1.3241e+00,(518, 518)).to(device)



    total_iters = args.epochs * len(dataloader)

    # previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
    #                  'log10': 100, 'silog': 100}




    for epoch in range(args.epochs):
        model.train()
        refinement_model.train()

        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                        print(f"Found NaN/Inf in {key}")

            images = batch['image'].to(device)
            gt_depth = batch['depth'].to(device).float()

            mask_valid = batch['mask'].to(device) if 'mask' in batch else torch.ones_like(gt_depth, dtype=torch.bool).to(device)
            ref_dirs = OF.get_camera_pixel_directions(images.shape[2:4], batch['n_intrinsics'], normalized_intrinsics=True).to(device)
            light_data = batch['light_data']  # Assumed to be [light_pos, light_dir, mu]
            light_pos, light_dir, mu = [d.to(device) for d in light_data]
            intrinsics = batch['n_intrinsics'].to(device)



            # Forward pass
            disparity, rgb_feats, colored_dot_product_feats, pred_normal = model(images, ref_dirs, light_pos, light_dir, mu, intrinsics)
            disp_preds = refinement_model(rgb_feats, colored_dot_product_feats, disparity)
            disp_preds = torch.clamp(disp_preds, min=1e-6, max=1e6)
            pred_depth = 1 / disp_preds
            pred_depth = torch.clamp(pred_depth, min=1e-6, max=1.0)
            pred_depth = pred_depth.unsqueeze(1)
            # ssi_loss, reg_loss, vnl_loss, pps_supp_loss, pps_corr_loss = criterion(images, pred_depth.unsqueeze(1), pred_normal, gt_depth, mask_valid, ref_dirs, light_pos, light_dir, mu, intrinsics)
            # total_loss = args.weight_ssi*ssi_loss + args.weight_reg*reg_loss + args.weight_vnl*vnl_loss + args.weight_pps_supp * pps_supp_loss + args.weight_pps_corr*pps_corr_loss
            mask_valid = mask_valid.bool()
            gt_depth = torch.clamp(gt_depth, min=1e-6, max=1.0)



            # silog_loss = silog_criterion(pred_depth, gt_depth, mask_valid)
            # gradl1_loss = gradl1_criterion(pred_depth, gt_depth, mask_valid)
            # loss = (args.weight_silog * silog_loss +
            #         args.weight_gradl1 * gradl1_loss)
            ssi_loss, reg_loss = midas_criterion(pred_depth, gt_depth, mask_valid)
            vnl_loss = vnl_criterion(pred_depth, gt_depth)
            ppssupp_loss = pps_supp_loss(pred_depth, gt_depth, ref_dirs, light_pos, light_dir, mu, intrinsics)

            loss = args.weight_ssi * ssi_loss + args.weight_reg * reg_loss + args.weight_vnl * vnl_loss + args.weight_pps_supp * ppssupp_loss


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(dataloader) + i
            # lr = args.lr * (1 - iters / total_iters) ** 0.9

            running_loss += loss.item() * images.size(0)

            iters = epoch * len(dataloader) + i

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr


            writer.add_scalar('train/loss', loss.item(), iters)
            # writer.add_scalar('train/ssi_loss', silog_loss.item(), iters)
            # writer.add_scalar('train/reg_loss', gradl1_loss.item(), iters)
            writer.add_scalar('train/ssi_loss', ssi_loss.item(), iters)
            writer.add_scalar('train/reg_loss', reg_loss.item(), iters)
            writer.add_scalar('train/vnl_loss', vnl_loss.item(), iters)
            writer.add_scalar('train/pps_supp_loss', ppssupp_loss.item(), iters)
            #writer.add_scalar('train/pps_corr_loss', pps_corr_loss.item(), iters)



            if i % 100 == 0:
                logger.info(
                    'Iter: {}/{}, LR: {:.7f}, total_loss: {:.3f}, ssi_loss: {:.3f}, reg_loss: {:.3f}, vnl_loss:{:.3f}, pps_supp_loss:{:.3f}'.format(
                        i, len(dataloader), lr, loss.item(), ssi_loss.item(), reg_loss.item(), vnl_loss.item(), ppssupp_loss.item()
                    )
                 )
                # logger.info(
                #     'Iter: {}/{}, LR: {:.7f}, total_loss: {:.3f}, SiLogLoss: {:.3f}, GradL1Loss: {:.3f}'.format(
                #         i, len(dataloader), lr, loss.item(), silog_loss.item(), gradl1_loss.item()
                #     )
                # )
                checkpoint = {
                    'model': model.state_dict(),
                    'refinement_model': refinement_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}_iter_{i}.pth"))

        model.eval()
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(),
                   'rmse': torch.tensor([0.0]).cuda(),
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(),
                   'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()

        for i, batch in enumerate(testloader):

            images = batch['image'].to(device)
            gt_depth = batch['depth'].to(device).float()
            intrinsics = batch['n_intrinsics'].to(device)
            ref_dirs = OF.get_camera_pixel_directions(images.shape[2:4], batch['n_intrinsics'],
                                                      normalized_intrinsics=True).to(device)
            light_data = [item.to(device) for item in batch['light_data']]
            light_pos, light_dir, mu = light_data

            with torch.no_grad():
                disparity, rgb_feats, colored_dot_product_feats, _ = model(images, ref_dirs, light_pos, light_dir, mu, intrinsics)
                disp_preds = refinement_model(rgb_feats, colored_dot_product_feats, disparity)
                pred = 1 / disp_preds
                pred = torch.clamp(pred, 0, 1)
                pred = pred.unsqueeze(1)


            mask_valid = batch['mask'].to(device) if 'mask' in batch else torch.ones_like(pred, dtype=torch.bool).to(
                device)
            valid_mask = (mask_valid == 1) & (pred >= 0) & (pred <= 1)

            if valid_mask.sum() < 10:
                continue


            cur_results = eval_depth(pred[valid_mask], gt_depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        if nsamples > 0:
            for k in results.keys():
                results[k] /= nsamples

        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
            *tuple([(v / nsamples).item() for v in results.values()])))
        logger.info('==========================================================================================')
        print()

        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

        # for k in results.keys():
        #     if k in ['d1', 'd2', 'd3']:
        #         previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        #     else:
        #         previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())


def main():
    parser = argparse.ArgumentParser(description="Train PPSNet Teacher Model")
    parser.add_argument("--data_dir", type=str, default="F:/monocular_depth_estimation_dataset/unity_endo", help="Path to C3VD dataset")
    parser.add_argument("--train_list", type=str, default="F:/monocular_depth_estimation_dataset/unity_endo/train.txt", help="Path to train list file")
    parser.add_argument("--test_list", type=str, default="F:/monocular_depth_estimation_dataset/unity_endo/test.txt",
                        help="Path to train list file")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to save logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--weight_ssi", type=float, default=1, help="loss weight ssi")
    parser.add_argument("--weight_reg",type=float, default=0.1, help="loss weight reg")
    parser.add_argument("--weight_vnl",type=float, default=10, help="loss weight ssi")
    parser.add_argument("--weight_pps_supp",type=float, default=0.1, help="loss weight pps_supp")
    parser.add_argument("--weight_pps_corr",type=float, default=0, help="loss weight pps_corr")
    parser.add_argument("--weight_silog", type=float, default=0, help="loss weight silog")
    parser.add_argument("--weight_gradl1", type=float, default=0, help="loss weight GradL1Loss")
    parser.add_argument("--save_path", type=str, default="./train_unity_v8")
    parser.add_argument("--pretrained_from", type=str, default="./train_unity_v7/checkpoint_epoch_5_iter_500.pth")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = Path(args.log_dir) / timestamp
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_teacher(args)


if __name__ == "__main__":
    main()