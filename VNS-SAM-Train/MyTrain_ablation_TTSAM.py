# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# CUDA_VISIBLE_DEVICES=1 PORT=101 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501  train_cod_wo_lowlevel_fusion.py  --checkpoint  ./pretrained_checkpoint/sam_vit_l_0b3195.pth  --model-type vit_l  --output work_dirs/hq_sam_l_wo_lowlevelfusion/ --batch_size_train 1
from optparse import check_builtin
import os
import argparse
from pickletools import uint8
from xml.dom.pulldom import default_bufsize
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader_edge import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
import einops
import math
import logging
from datetime import datetime
from torch import distributed as dist
# from models import MaskDecoderHQ, MaskDecoderEnhance, model_registry
from datasets import dataset_registry
import py_sod_metrics
from tqdm import tqdm



class MaskDecoderTT(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                                multi_stage=False,
                                return_attnmap=False,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h", "vit_t"]
        
        
        checkpoint_dict = {"vit_b":"pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth",
                           'vit_t': "pretrained_checkpoint/mobile_sam_maskdecoder.pth"}
        
        checkpoint_path = checkpoint_dict[model_type]
        
        self.load_state_dict(torch.load(checkpoint_path), strict=False)
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            if "output_hypernetworks" in n or "mask_token" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],     
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],       
            )
            
            masks.append(mask)
            # dense_preds.append(dense_pred)
            # inter_masks.append(inter_mask)
            iou_preds.append(iou_pred)
            # attn_maps.append(attn_map)
            
        masks = torch.cat(masks, 0)
        # dense_preds = torch.cat(dense_preds, 0)
        # inter_masks = torch.cat(inter_masks, 0)
        iou_preds = torch.cat(iou_preds, 0)


        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks



def evaluate_metrics(args):
    # metrics
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    
    mask_dataset_list = ['CHAMELEON','CAMO', 'COD10K', 'NC4K']
    # mask_name_list = sorted(os.listdir(args.mask_dir))
    results_list = []
    
    pred_dir = args.pred_dir
    for i, dataset in enumerate(mask_dataset_list):
        # metrics
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure()
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()
        print(f"================{datetime.now()} [{i}] Processing {dataset}================")
        logging.info(f"[{i}] Processing {dataset}")
        mask_dataset_dir = os.path.join(args.mask_dir, dataset)
        pred_dataset_dir = os.path.join(pred_dir, dataset)
        pred_name_list = sorted(os.listdir(pred_dataset_dir))
        for j in tqdm(range(len(pred_name_list))):
            # print(f"[{i}] Processing {mask_name}...")

            mask_name = pred_name_list[j]

            mask_path = os.path.join(mask_dataset_dir+'/GT', mask_name)
            pred_path = os.path.join(pred_dataset_dir, mask_name)        
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)
        
        
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]

        curr_results = {
            "MAE": mae,
            "Smeasure": sm,
            "wFmeasure": wfm,
            # E-measure for sod
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max()
        }
        
        results_list.append(curr_results)
        print(curr_results)
        # logging.info(str(curr_results))
   
    return results_list


def generate_benchmark_table(results):
    latex = []
    for k in range(len(results)):
        result = results[k]
        # print(line.split('Model:')[1].split(') Smeasure')[0], model_lst[i])
        S_measure = '%.3f'%result['Smeasure']
        w_F = '%.3f'%result['wFmeasure']
        mean_E_m = '%.3f'%result['meanEm']
        MAE = round(result['MAE'],3)
        
        res_latex = '& {}   & {}   & {}   & {} '.format(S_measure, mean_E_m, w_F, MAE)
        latex.append(res_latex)
        print(res_latex, end='\n')
    return latex
       


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask) in enumerate(masks):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def heatmap(x_show, img, name=None):
    x_show = torch.mean(x_show, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)
    if img is not None:
        img = img.data.cpu().numpy().squeeze()
        # print(img.shape)
        img = img.transpose((1, 2, 0))

        # img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        img = img[:, :, ::-1]
        # print(img.shape)
        # exit()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.uint8(255 * img)
        img = cv2.resize(img, (1024, 1024))
    x_show = np.uint8(255 * x_show)
    x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)
    x_show = cv2.resize(x_show, (1024, 1024))
    
    # print(x_show.shape, img.shape)
    if img is not None:
        x_show = cv2.addWeighted(img, 0.6, x_show, 0.4, 0)
        # x_show = img
    if name is not None:
        cv2.imwrite( name, x_show)
    
    # cv2.imshow('img', x_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def show_saliency_map(x_show, img, name=None):
    x_show = x_show.data.cpu().numpy().squeeze()
    x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)
    if img is not None:
        img = img.data.cpu().numpy().squeeze()
        # print(img.shape)
        img = img.transpose((1, 2, 0))

        # img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        img = img[:, :, ::-1]
        # print(img.shape)
        # exit()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.uint8(255 * img)
        img = cv2.resize(img, (320, 320))
    x_show = np.uint8(255 * x_show)
    x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)
    x_show = cv2.resize(x_show, (320, 320))
    
    # print(x_show.shape, img.shape)
    if img is not None:
        x_show = cv2.addWeighted(img, 0.6, x_show, 0.4, 0)
        # x_show = img
    if name is not None:
        cv2.imwrite( name, x_show)
    # cv2.imshow('img', x_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def visualize_attention_map_opencv(attention_weights, img, name=None):
    
    if img is not None:
        img = img.data.cpu().numpy().squeeze()
        # print(img.shape)
        img = img.transpose((1, 2, 0))
        # img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        img = img[:, :, ::-1]
        # print(img.shape)
        # exit()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.uint8(255 * img)
        img = cv2.resize(img, (1024, 1024))
        
    # 将张量移动到 CPU
    attention_weights_cpu = attention_weights

    # 归一化到 [0, 255] 的范围
    normalized_attention = (attention_weights_cpu - attention_weights_cpu.min()) / (attention_weights_cpu.max() - attention_weights_cpu.min()) * 255

    # 使用 OpenCV 绘制热力图
    heatmap = cv2.applyColorMap(np.uint8(normalized_attention), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (1024, 1024))
    
    if img is not None:
        x_show = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
        # x_show = img
    if name is not None:
        cv2.imwrite(name, x_show)
        
    # 可视化
    # cv2.imshow('Attention Map', heatmap)
    # cv2.imwrite('work_dirs/sam-l-edge-enhancev4-23-cod/1.jpg', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument("--model", type=str, default='sam-enhance',
                        help="select model, such as 'sam-hq', 'sam-en', etc..")
    parser.add_argument("--dataset", type=str, default="COD", 
                        help="select train and val dataset, such as 'COD', 'nosaliency', etc. ")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--weight', type=float, default=0.5, help='loss weight')
    
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument("--prompt", type=str, default='box',
                        help="the style of prompt")
    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):
    val_set_name = [i["name"] for i in valid_datasets]
    train_set_name = [i["name"] for i in train_datasets]
    
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module


    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        
        if args.restore_model:
            if dist.get_rank() == 0:
                print("restore model from:", args.restore_model)  
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            elif dist.get_rank() == 1:
                net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location={'cuda:0':'cuda:1'}))
            elif dist.get_rank() == 2:
                net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location={'cuda:0':'cuda:2'}))
            elif dist.get_rank() == 3:
                net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location={'cuda:0':'cuda:3'}))
                
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, val_set_name, lr_scheduler)
        # # test
        # test_stats = evaluate(args, net, sam, valid_dataloaders, val_set_name, save_preds=False)

    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
        # if misc.is_main_process():
        
        evaluate(args, net, sam, valid_dataloaders, val_set_name, True, args.visualize)



def dice_loss_edge(predict, target):
    if predict.shape != target.shape:
        predict = F.interpolate(predict, size=target.shape[2:], mode='bilinear', align_corners=False)
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    if pred.shape != mask.shape:
        pred = F.interpolate(pred, size=mask.shape[2:], mode='bilinear', align_corners=False)
    
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, val_set_name, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    # checkpoint= torch.load(args.checkpoint)
    # decoder_checkpoint = {}
    # for k, v in checkpoint.items():
    #     if "mask_decoder" in k:
    #         decoder_checkpoint[k.replace("mask_decoder.", "")] = v 
    # torch.save(decoder_checkpoint, './pretrained_checkpoint/mobile_sam_maskdecoder.pth')
    # exit()
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start, epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        if dist.get_rank() == 0:
            logging.info("{}  [epoch] {}  [learning rate] {}" .format(datetime.now(), epoch, optimizer.param_groups[0]["lr"]))

        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,100):
            inputs, labels, edges = data['image'], data['label'], data['edge']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                edges = edges.cuda()
            
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_retrain = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            loss_mask1, loss_dice1 = loss_masks(masks_retrain, labels/255.0, len(masks_retrain))
            loss = loss_mask1 + loss_dice1


            # loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge": loss_edge}
            loss_dict = {"loss_mask1": loss_mask1, "loss_dice1":loss_dice1
                         }

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        if dist.get_rank() == 0:
            logging.info('Averaged states:'.format(metric_logger))
        
        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)

        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        lr_scheduler.step()
        net.train()  

    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)
    test_stats = evaluate(args, net, sam, valid_dataloaders, val_set_name, save_preds=False)

def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, net, sam, valid_dataloaders, val_set_name, save_preds=False, visualize=False):
    net.eval()
    print("Validating...")
    if dist.get_rank() == 0:
        logging.info('Start Validating...')
        
    test_stats = {}
    results = []
    for k in range(len(valid_dataloaders)):
        
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure()
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()
        
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))
        if dist.get_rank() == 0:
            logging.info("valid dataloader len: {}".format(len(valid_dataloader)))
        
        save_dir = os.path.join(args.output, val_set_name[k])
        
        for data_val in metric_logger.log_every(valid_dataloader, 100):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori, ori_gt_path = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['ori_gt_path']
            img_name = ori_gt_path[0].split('/')[-1]
            
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            

            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            labels_points = misc.masks_sample_points(labels_val[:,0,:,:], k=3)
            input_keys = [args.prompt]

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_retrain = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

           
            
            # vis heatmap
            # os.makedirs(args.output, exist_ok=True)
            # os.makedirs(save_dir, exist_ok=True)
            # heatmap(F.relu(mask_feats), inputs_val.squeeze(0), os.path.join(save_dir, img_name))
            # show_saliency_map(mask_feats, inputs_val.squeeze(0), os.path.join(save_dir, img_name))
            
            # vis attntion map
            # vis_attn_map = F.softmax(attn_maps.mean(1), dim=-1)[0, -4, :].squeeze().reshape(64,64)
            # vis_attn_map = attn_maps[0, 2, :, -3].squeeze().reshape(64,64)
            # visualize_attention_map_opencv(vis_attn_map.detach().cpu().numpy(), inputs_val.squeeze(0), os.path.join(save_dir, img_name))
            # continue

            
            
            iou = compute_iou(masks_retrain, labels_ori)
            boundary_iou = compute_boundary_iou(masks_retrain,labels_ori)

            res = F.upsample(masks_retrain, size=tuple(labels_ori.shape[2:]), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = np.uint8(res * 255)
            mask = np.uint8(labels_ori.data.cpu().numpy().squeeze().squeeze())

            WFM.step(pred=res, gt=mask)
            SM.step(pred=res, gt=mask)
            EM.step(pred=res, gt=mask)
            MAE.step(pred=res, gt=mask)
            
            
            # print(mask_feats.shape, inputs_val.shape)
            # save predictions
            if save_preds:
                os.makedirs(args.output, exist_ok=True)
                os.makedirs(save_dir, exist_ok=True)
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    res = F.upsample(masks_retrain, size=tuple(labels_ori.shape[2:]), mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    cv2.imwrite(os.path.join(save_dir, img_name), res*255)
            
            if visualize:
                # print("visualize")
                os.makedirs(args.output, exist_ok=True)
                os.makedirs(save_dir, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_retrain.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    # print('base:', base)
                    save_base = os.path.join(save_dir, img_name)
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor(0)
                    show_boundary_iou = torch.tensor(0)
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)   

            # if visualize:
            #     print("visualize")
            #     os.makedirs(args.output, exist_ok=True)
            #     masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
            #     for ii in range(len(imgs)):
            #         base = data_val['imidx'][ii].item()
            #         print('base:', base)
            #         save_base = os.path.join(args.output, str(k)+'_'+ str(base))
            #         imgs_ii = imgs[ii].astype(dtype=np.uint8)
            #         show_iou = torch.tensor(0)
            #         show_boundary_iou = torch.tensor(0)
            #         show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)   

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
        
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]

        curr_results = {
            "MAE": mae,
            "Smeasure": sm,
            "wFmeasure": wfm,
            # E-measure for sod
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max()
        }
        results.append(curr_results)
        
        print('============================ {} ================'.format(val_set_name[k]))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print(curr_results)
        print("Averaged stats:", metric_logger)
        if dist.get_rank() == 0:
            logging.info('==Dataset: {}===  Averaged stats: {}'.format(val_set_name[k], metric_logger))
            logging.info(str(curr_results))
            
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
    latex = generate_benchmark_table(results)
    if dist.get_rank() == 0:
        logging.info(latex)
    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------
    args = get_args_parser()
    train_datasets, valid_datasets = dataset_registry[args.dataset]()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(filename=args.output + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # net = model_registry[args.model](args.model_type) 
    net = MaskDecoderTT(args.model_type)
    model_total_params = sum(p.numel() for p in net.parameters())
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), 'model_total_params:' + str(model_total_params))
    # save checkpoint 
    sam_ckpt = torch.load(args.checkpoint)
    hq_decoder = torch.load('/home/ps/Guo/Project/sam-hq-main/train/work_dirs/ablation/TTSAM/epoch_11.pth')
    for key in hq_decoder.keys():
        sam_key = 'mask_decoder.'+key
        # if sam_key not in sam_ckpt.keys():
        sam_ckpt[sam_key] = hq_decoder[key]
    model_name = "/sam_hq_epoch_"+str(11)+".pth"
    torch.save(sam_ckpt, args.output + model_name)
    exit()
    
    main(net, train_datasets, valid_datasets, args)


