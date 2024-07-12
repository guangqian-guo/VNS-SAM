"""v6
TokenInter attention downsample rate 1

"""

"""
token 交互使用一个交叉注意力层。
2024/07/12
--guoguangqian
"""

import os
import argparse
from turtle import forward
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
from .neck import TFD
from utils.dataloader_edge import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
import einops
import math


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

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate   # 128
        self.num_heads = num_heads  # 8
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)   # 1 8 4096 16

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        # print(attn.shape)

        attn = attn / math.sqrt(c_per_head)
        attn = torch.sigmoid(attn)

        # Get output
        out = attn @ v

        out = self._recombine_heads(out)  # 1 4096 128

        out = self.out_proj(out)

        return out


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
                nn.ReLU(),
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU()
            )
        
        self.edge_pred_layer1 = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=1
                ),
            nn.ReLU()
        )
        
        self.edge_pred_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=1),
        )
        
        # feat layer
        self.hqfeat_layer0 = nn.Sequential(
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU()
            )
        
        
        self.hqfeat_layer1 = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    inner_channels,
                    kernel_size=1
                ),
            nn.ReLU()
        )

        self.hqfeat_head = nn.Sequential(
            BasicConv2d(
                    inner_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
            nn.ReLU()
        )

        
        self.up_sample_layers = nn.ModuleList()
        # assert up_sample_scale == 4

        self.up_sample_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm2d(out_channels),
                nn.GELU(), 
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm2d(out_channels),
                nn.GELU(), 
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
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



# class Token_Attn(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.q_proj = nn.Linear(dim, dim // 2)
#         self.k_proj = nn.Linear(dim, dim // 2)
#         self.v_proj = nn.Linear(dim, dim // 2)
#         self.out_proj = nn.Linear(dim // 2, dim)
    
#     def forward(self, t1, t2):
#         q = self.q_proj(t1)
#         k = self.k_proj(t2)
#         v = self.v_proj(t2)
        

#         return 



class TokenInter(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.f1 = nn.Linear(dim * 2, dim)
        self.token_attn1 = Attention(dim, 8, 2)
        self.token_attn2 = Attention(dim, 8, 2) 


    def forward(self, mask_token, edge_token):
        fusion_token = torch.cat([mask_token, edge_token], dim=1)
        fusion_token = self.f1(fusion_token)    # 1 256

        # mask toekn
        # 1 1 256
        mask_token_o = self.token_attn1(fusion_token.unsqueeze(0), mask_token.unsqueeze(0), mask_token.unsqueeze(0))

        # edge token
        # 1 1 256
        edge_token_o = self.token_attn2(fusion_token.unsqueeze(0), edge_token.unsqueeze(0), edge_token.unsqueeze(0))

        return mask_token_o.squeeze(0), edge_token_o.squeeze(0)        


class MaskDecoderEnhancefusionTIv6(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                                multi_stage=True,
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
            p.requires_grad = False
            if "internal" in n:
                print("new added params:", n)
                p.requires_grad = True
            
        
        transformer_dim=256
        vit_dim_dict = {"vit_b":[768]*4,"vit_l":[1024] * 4,"vit_h":[1280]*4, "vit_t": [160, 320, 320]}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(2, transformer_dim)  # 一个mask token， 一个edge token
        
        self.hf_mlp = (MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3))
        
        self.edge_mlp = (MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3))

        self.num_mask_tokens = self.num_mask_tokens + 2   #NOTE:  之前都设置错了，都是 +1，导致mask token好像根本没有学到。--2023.12.11
        
        # self.neck = SAMAggregatorNeck(in_channels=vit_dim)
        self.neck = TFD(in_channels=vit_dim)
        
        self.embedding_maskfeature = (
            nn.Sequential(
                nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                LayerNorm2d(transformer_dim // 4),
                nn.GELU(),
                nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
                nn.GELU())
        )
        
        self.embedding_edgefeature = (
            nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                nn.GELU()
            )
        )


        # token interactive
        self.ti_layer = TokenInter(transformer_dim)



        # dense pred
        self.dense_attn = Attention(transformer_dim, 8, 2)
       
        self.dense_pred_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            # nn.Sigmoid(),         ######################### NOTE: there is no sigmid before --3.5
            nn.Linear(transformer_dim, 1),
            # nn.Sigmoid()        ######################### NOTE: there is no sigmid before --3.5
        )
        
        self.fusion_layer = nn.Sequential(
                nn.Conv2d(2, 256, 3, 1, 1), 
                nn.GELU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(256, 1, 3, 1, 1))

    
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
        dense_preds = []
        fusion_preds = []
        inter_masks = []
        iou_preds = []
        mask_feats = []  # for vis
        attn_maps = [] # for vis
        for i_batch in range(batch_len):
            mask, inter_mask, dense_pred, fusion_mask, iou_pred, mask_feat, attn_map = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0),
                edge_feat = edge_pred[i_batch].unsqueeze(0),
            )
            
            masks.append(mask)
            dense_preds.append(dense_pred)
            inter_masks.append(inter_mask)
            iou_preds.append(iou_pred)
            mask_feats.append(mask_feat)
            attn_maps.append(attn_map)
            fusion_preds.append(fusion_mask)
            
        masks = torch.cat(masks, 0)
        dense_preds = torch.cat(dense_preds, 0)
        
        inter_masks = torch.cat(inter_masks, 0)
        iou_preds = torch.cat(iou_preds, 0)
        mask_feats = torch.cat(mask_feats, 0)
        fusion_preds = torch.cat(fusion_preds)
        # attn_maps = torch.cat(attn_maps, 0)   # only support batch=1
        
        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-2, self.num_mask_tokens-1), :, :]
        edge_pred = torch.sigmoid(masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :])
        inter_masks_hq = inter_masks[:, slice(0, 1), :, :]
        inter_edge_pred = torch.sigmoid(inter_masks[:, slice(1, 2), :, :])
        masks_dense = dense_preds[:, slice(0,1), :, :]
        fusion_preds = fusion_preds[:, slice(0,1), :, :]


        if hq_token_only:
            return masks_hq, edge_pred, inter_masks_hq, inter_edge_pred, masks_dense, fusion_preds
        else:
            return masks_sam, masks_hq, edge_pred, inter_masks_hq, inter_edge_pred, masks_dense, fusion_preds, mask_feats, attn_maps



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
        image_embeddings = image_embeddings 
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        
        # Run the transformer
        hs, src, attn_map = self.transformer(src, pos_src, tokens)
        internal_hs = hs[:-1][0]
        hs = hs[-1]
        internal_src = src[:-1][0]
        src = src[-1]

        iou_token_out = hs[:, 0, :]  
        mask_tokens_out = hs[:, 1 : self.num_mask_tokens, :]   #NOTE： 这里也设置的不对，应该就是self.num_mask_tokens，之前是2+self.num_mask_tokens  2023.12.11
        
        # dense pred
        hq_token = (mask_tokens_out[:, -2, :] + mask_tokens_out[:, -1, :]).unsqueeze(1)        
        feat = src
        dense_pred = self.dense_attn(feat, hq_token, hq_token) + feat # 1 4096 256
        dense_pred = self.dense_pred_head(dense_pred)
        dense_pred = dense_pred.transpose(1, 2).view(b, 1, h, w)
        dense_ = torch.cat([dense_pred], dim=1)
        
        
        #----------------------------------------------------------------
        
        inter_mask_tokens_out = internal_hs[:, 1 : self.num_mask_tokens, :] 

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        internal_src = internal_src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = (self.embedding_maskfeature[-1](upscaled_embedding_sam) + hq_feature) # b 32 256 256

        upscaled_inter_embedding_ours = self.embedding_maskfeature(self.output_upscaling(internal_src)) + hq_feature
        
        # vis ------------------------------
        # heatmap(upscaled_embedding_ours, None)
        #-----------------------------------
        
        upscaled_edge_embedding = self.embedding_edgefeature(edge_feat)
        upscaled_inter_edge_embedding = self.embedding_edgefeature(edge_feat)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            # elif i < 5:
            #     hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :] + mask_tokens_out[:, i+1, :]))
            # else:
            #     hyper_in_list.append(self.edge_mlp(mask_tokens_out[:, i, :]))
        mask_hq_token, edge_token = self.ti_layer(mask_tokens_out[:, -2, :], mask_tokens_out[:, -1, :])
        edge_token = self.edge_mlp(edge_token)
        mask_hq_token = self.hf_mlp(mask_hq_token)  # b 32


        inter_mask_hq_token, inter_edge_token = self.ti_layer(inter_mask_tokens_out[:, -2, :], inter_mask_tokens_out[:, -1, :])
        inter_edge_token = self.edge_mlp(inter_edge_token)
        inter_mask_hq_token = self.hf_mlp(inter_mask_hq_token)
        

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)

        masks_ours = (mask_hq_token @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # 和mask token做点乘，作为输出的mask。

        edge_pred = (edge_token @ upscaled_edge_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        inter_masks_ours = (inter_mask_hq_token @ upscaled_inter_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        inter_edge = (inter_edge_token @ upscaled_inter_edge_embedding.view(b, c, h * w)).view(b, -1, h, w)

        masks = torch.cat([masks_sam, masks_ours, edge_pred],dim=1)
        inter_masks  = torch.cat([inter_masks_ours, inter_edge], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        # results fusion   
        dense_ = F.interpolate(dense_, size = masks_ours.shape[2:], mode='bilinear', align_corners=False)
        fusion_mask = torch.cat((masks_ours, dense_), dim=1)
        fusion_mask = self.fusion_layer(fusion_mask)
        

        
        return masks, inter_masks, dense_, fusion_mask, iou_pred, edge_pred, attn_map
