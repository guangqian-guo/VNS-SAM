# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# CUDA_VISIBLE_DEVICES=1 PORT=101 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501  train_cod_wo_lowlevel_fusion.py  --checkpoint  ./pretrained_checkpoint/sam_vit_l_0b3195.pth  --model-type vit_l  --output work_dirs/hq_sam_l_wo_lowlevelfusion/ --batch_size_train 1
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader_edge import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
import einops

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
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            in_channels=[1024]*4,
            inner_channels=256,
            out_channels=32,
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            up_sample_scale=4,
            init_cfg=None,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels
        self.stride = stride
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.down_sample_layers.append(
                nn.Sequential(
                    BasicConv2d(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                    ),
                    nn.ReLU()
                )
            )

        self.fusion_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.fusion_layers.append(
                nn.Sequential(BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU()
                )
            )

        # edge layer
        self.edge_pred_layer0 = nn.Sequential(
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(),
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU()
            )
        self.edge_pred_layer1 = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=1
                ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU()
        )
        
        self.edge_pred_head = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=1
                )
        )
        
        # feat layer
        self.hqfeat_layer0 = nn.Sequential(
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(),
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU()
            )
        
        self.hqfeat_layer1 = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=1
                ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU()
        )

        self.hqfeat_head = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1
                ),
            nn.ReLU()
        )

        
        self.up_sample_layers = nn.ModuleList()
        # assert up_sample_scale == 4

        self.up_sample_layers.append(
            nn.Sequential(
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2),
            LayerNorm2d(inner_channels),
            nn.GELU(),
            nn.ConvTranspose2d(inner_channels, out_channels, kernel_size=2, stride=2),
            nn.GELU(),
            )
        )


        
    
    def forward(self, inputs):
        inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in range(len(self.in_channels))]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]
        
        x = None
        pyramid_feats = []
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
            # pyramid_feats.append(x)
        
        # edge pred
        edge_feat = self.edge_pred_layer0(x) + x
        edge_feat = self.edge_pred_layer1(edge_feat)
        edge_feat = self.edge_pred_head(edge_feat)

        # high quality features
        img_feat = self.hqfeat_layer0(x) + x
        img_feat = self.hqfeat_layer1(img_feat)
        img_feat = self.hqfeat_head(img_feat)
        img_feat = self.up_sample_layers[0](img_feat)

        return img_feat, edge_feat


class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b":"pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(2, transformer_dim)  # 一个mask token， 一个edge token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.edge_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.num_mask_tokens = self.num_mask_tokens + 1
        
        self.neck = SAMAggregatorNeck()

        # self.compress_vit_feat = nn.Sequential(
        #                                 nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
        #                                 LayerNorm2d(transformer_dim),
        #                                 nn.GELU(), 
        #                                 nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        # self.embedding_encoder = nn.Sequential(
        #                                 nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #                                 LayerNorm2d(transformer_dim // 4),
        #                                 nn.GELU(),
        #                                 nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #                             )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        
        self.embedding_edgefeature = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,  # length=4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """

        # vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT

        hq_features, edge_pred = self.neck(interm_embeddings)    # 1 32 256 256, 1 1 256 256

        # hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)   # 做了一个深层和浅层的特征融合
        
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],     
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],     
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],       
                hq_feature = hq_features[i_batch].unsqueeze(0),
                edge_feat = edge_pred[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-2, self.num_mask_tokens-1), :, :]
        edge_pred = torch.sigmoid(masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :])
        
        if hq_token_only:
            return masks_hq, edge_pred
        else:
            return masks_sam, masks_hq, edge_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        image_embeddings = image_embeddings + edge_feat
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (2 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        # print(self.embedding_maskfeature(upscaled_embedding_sam).shape)   # 1 32 256 256 
        upscaled_embedding_ours = (self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature)
        upscaled_edge_embedding = self.embedding_edgefeature(edge_feat)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            # elif i < 5:
            #     hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :] + mask_tokens_out[:, i+1, :]))
            # else:
            #     hyper_in_list.append(self.edge_mlp(mask_tokens_out[:, i, :]))
        
        edge_token = self.edge_mlp(mask_tokens_out[:, -1, :])
        mask_hq_token = self.hf_mlp(mask_tokens_out[:, -2, :] + mask_tokens_out[:, -1, :])

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (mask_hq_token @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # 和mask token做点乘，作为输出的mask。
        edge_pred = (edge_token @ upscaled_edge_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        masks = torch.cat([masks_sam, masks_ours, edge_pred],dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)
        
        return masks, iou_pred

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
        logging.info(str(curr_results))
   
    return results_list

def generate_benchmark_table(results, args):
    results_log = './res/{}/results.txt'.format(args.pred_dir.split('/')[-1])
    
    for k in range(len(results)):
        result = results[k]
        # print(line.split('Model:')[1].split(') Smeasure')[0], model_lst[i])
        S_measure = '%.3f'%result['Smeasure']
        w_F = '%.3f'%result['wFmeasure']
        mean_E_m = '%.3f'%result['meanEm']
        MAE = round(result['MAE'],3)
        res_latex = '& {}   & {}   & {}   & {}'.format(S_measure, mean_E_m, w_F, MAE)
        print(res_latex, end='\n')
        with open(results_log, 'a') as f:
            f.write(res_latex+'\n')


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

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):

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
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
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
    
        evaluate(args, net, sam, valid_dataloaders, True, args.visualize)


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


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start, epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
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

            masks_hq, edge_pred = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))

            loss_edge = dice_loss_edge(edge_pred, edges)   # TODO !!!
            
            loss = loss_mask + loss_dice + loss_edge
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge": loss_edge}

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
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        if epoch == epoch_num-1:
            test_stats = evaluate(args, net, sam, valid_dataloaders, save_preds=True)
        
        # train_stats.update(test_stats)
            
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
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

def evaluate(args, net, sam, valid_dataloaders, save_preds=False, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}
    dataset_name = ['CHAMELEON','CAMO', 'COD10K', 'NC4K']
    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        save_dir = os.path.join(args.output, dataset_name[k])
        os.makedirs(save_dir, exist_ok=True)
        
        for data_val in metric_logger.log_every(valid_dataloader, 100):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori, ori_gt_path = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['ori_gt_path']
            img_name = ori_gt_path[0].split('/')[-1]
            
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            

            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
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
            
            masks_sam, masks_hq, edge_pred = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            
            # iou = compute_iou(masks_hq,labels_ori)
            # boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
            
            # save predictions
            if save_preds:
                os.makedirs(args.output, exist_ok=True)
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    res = F.upsample(masks_hq, size=tuple(labels_ori.shape[2:]), mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    cv2.imwrite(os.path.join(save_dir, img_name), res*255)


            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(edge_pred.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor(0)
                    show_boundary_iou = torch.tensor(0)
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                    

            # loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            # loss_dict_reduced = misc.reduce_dict(loss_dict)
            # metric_logger.update(**loss_dict_reduced)


        # print('============================')
        # # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        # resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        # test_stats.update(resstat)


    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    # added by guo
    dataset_cod = {"name": "COD",
                    "im_dir": "./data/COD/TrainDataset/Imgs",
                    "gt_dir": "./data/COD/TrainDataset/GT",
                    "edge_dir": "./data/COD/TrainDataset/Edge",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    # valid set
    dataset_camo_val = {"name": "CAMO",
                    "im_dir": "./data/COD/TestDataset/CAMO/Imgs",
                    "gt_dir": "./data/COD/TestDataset/CAMO/GT",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}
    dataset_cha_val = {"name": "CHAMELEON",
                    "im_dir": "./data/COD/TestDataset/CHAMELEON/Imgs",
                    "gt_dir": "./data/COD/TestDataset/CHAMELEON/GT",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}
    dataset_cod10k_val= {"name": "COD10K",
                    "im_dir": "./data/COD/TestDataset/COD10K/Imgs",
                    "gt_dir": "./data/COD/TestDataset/COD10K/GT",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}
    dataset_NC4K_val = {"name": "NC4K",
                    "im_dir": "./data/COD/TestDataset/NC4K/Imgs",
                    "gt_dir": "./data/COD/TestDataset/NC4K/GT",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 

    # guo
    train_datasets=[dataset_cod]
    valid_datasets = [dataset_cha_val, dataset_camo_val,  dataset_cod10k_val, dataset_NC4K_val]
     
    args = get_args_parser()
    net = MaskDecoderHQ(args.model_type) 

    main(net, train_datasets, valid_datasets, args)
