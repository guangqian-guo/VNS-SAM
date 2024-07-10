# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# add DWC, 2023.12.16 --ggq


import enum
import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock
import numpy as np
from einops import rearrange


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        add_dwc: bool = False, 
        multi_stage: bool = False,     # 在decoder 添加多阶段损失。
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        
        self.add_dwc = add_dwc    # add by guo
        self.multi_stage = multi_stage   # added by guo
        
        for i in range(depth):
            # self.layers.append(
            #     TwoWayAttentionBlock(
            #         embedding_dim=embedding_dim,
            #         num_heads=num_heads,
            #         mlp_dim=mlp_dim,
            #         activation=activation,
            #         attention_downsample_rate=attention_downsample_rate,
            #         skip_first_layer_pe=(i == 0),
            #         add_dwc=True,
            #     ) 
            # ) if self.add_dwc else  self.layers.append(
            #                             TwoWayAttentionBlock(
            #                                 embedding_dim=embedding_dim,
            #                                 num_heads=num_heads,
            #                                 mlp_dim=mlp_dim,
            #                                 activation=activation,
            #                                 attention_downsample_rate=attention_downsample_rate,
            #                                 skip_first_layer_pe=(i == 0),
            #                             ) 
            #                         )
            self.layers.append(
                                        TwoWayAttentionBlock(
                                            embedding_dim=embedding_dim,
                                            num_heads=num_heads,
                                            mlp_dim=mlp_dim,
                                            activation=activation,
                                            attention_downsample_rate=attention_downsample_rate,
                                            skip_first_layer_pe=(i == 0),
                                        ) 
                                    )
        if self.multi_stage:
            self.internal_attn_token_to_image = Attention(
                embedding_dim, num_heads, downsample_rate=attention_downsample_rate
                )
            self.norm_internal_attn = nn.LayerNorm(embedding_dim)


        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # self.final_attn_token_to_image = Attention_t2i_dwc(
        #     embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        # )  if self.add_dwc else Attention(
        #                                 embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        #                             )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        return_attn=False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        # Apply transformer blocks and final layernorm
        multi_queries = []
        multi_keys = []
        for i, layer in enumerate(self.layers):
            queries, keys = layer(  # self attn, token2img img2token
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

           
            multi_queries.append(queries)
            multi_keys.append(keys)
        
        # Apply the final attenion layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out, _ = self.final_attn_token_to_image(q=q, k=k, v=keys)  # _ is attn map
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        multi_queries.append(queries)
        multi_keys.append(keys)
        
        if self.multi_stage:
            return multi_queries, multi_keys
        else:
            return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        add_dwc: bool = False,        # added by guo
    ) -> None:
        
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.add_dwc = add_dwc
        self.internal_dim = embedding_dim // attention_downsample_rate   # 128
        self.num_heads = num_heads  # 8
        kernelsize = 5
        
        
        self.cross_attn_token_to_image = Attention_t2i_dwc(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        )  if self.add_dwc else Attention(
                                        embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
                                    )
        
        # self.cross_attn_token_to_image = Attention(
        #                                 embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        #                             )
        
        # self.cross_attn_token_to_image = Focus_Attention(
        #     embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        # )  if self.add_dwc else Attention(
        #                                 embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        #                             )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention_i2t_dwc(
                                        embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
                                    ) if self.add_dwc else Attention(
                                        embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
                                    )
        
        # self.cross_attn_image_to_token = Focus_Attention(
        #                                 embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        #                             ) if self.add_dwc else Attention(
        #                                 embedding_dim, num_heads, downsample_rate=attention_downsample_rate,
        #                             )
        

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, _ = self.self_attn(q=queries, k=queries, v=queries)
        
        else:
            q = queries + query_pe
            attn_out, _ = self.self_attn(q=q, k=q, v=queries)
            
            # vis -------------------------------------------
            # vis_attn_map = _[0,0,:,:].squeeze()
            # visualize_attention_map(vis_attn_map.detach().cpu().numpy())
            # ----------------------------------------------
            queries = queries + attn_out
        
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_token_to_image(q=q, k=k, v=keys)

        
        #  vis -----------------------------------------------------
        # vis_attn_map = _[0,0,-1,:].squeeze().reshape(64,64)
        # visualize_attention_map_opencv(vis_attn_map.detach().cpu().numpy())
        #------------------------------------------------------------
        
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


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
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v

        out = self._recombine_heads(out)  # 1 4096 128

        out = self.out_proj(out)

        return out, attn

class Attention_dwc(nn.Module):   # v18
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        add_dwc: bool = True,
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

        if add_dwc:
            kernelsize = 5
            self.dwc = nn.Conv2d(in_channels=self.internal_dim // self.num_heads, out_channels=self.internal_dim // self.num_heads, kernel_size=kernelsize,
                             groups=self.internal_dim // self.num_heads, padding=kernelsize // 2)
    
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
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v      # 1 8 9 16


        #------------------------ add dwc ----------------------------------------------------#
        b, n_heads, n_tokens, c_per_head = v.shape
        resolution = int(n_tokens ** 0.5)
        feature_map = v.reshape(-1, n_tokens, c_per_head).permute(0, 2, 1)  # 8 16 4096
        feature_map = rearrange(feature_map, "b c (h w) -> b c h w", h=resolution, w=resolution)  # 8 16 64 64
        feature_map = self.dwc(feature_map)           # 8 16 64 64
        feature_map = nn.AdaptiveAvgPool2d((1,1))(feature_map)    # 8 16 1 1
        feature_map = rearrange(feature_map, "b c h w -> b (h w) c")  # 8 1 1 16
        feature_map = feature_map.reshape(b, n_heads, -1, c_per_head)  # 1 8 1 16
        out = out + feature_map
        #------------------------------------------------------------------------------------#


        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out, attn


class Attention_t2i_dwc(nn.Module):  #v18-2
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        add_dwc: bool = True,
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

        if add_dwc:
            kernelsize = 5
            self.dwc = nn.Conv2d(in_channels=self.internal_dim // self.num_heads, out_channels=self.internal_dim // self.num_heads, kernel_size=kernelsize,
                             groups=self.internal_dim // self.num_heads, padding=kernelsize // 2)
    
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
        attn = torch.softmax(attn, dim=-1)   # 9 4096
        
        # vis
        # print(attn.shape)
        # visualize_attention_maps(rearrange(attn[0][0], "c (h w) -> c h w", h=64, w=64), 3)
        
        #------------------------ add dwc ----------------------------------------------------#
        b, n_heads, n_tokens, c_per_head = v.shape
        resolution = int(n_tokens ** 0.5)
        feature_map = v.reshape(-1, n_tokens, c_per_head).permute(0, 2, 1)  # 8 16 4096
        feature_map = rearrange(feature_map, "b c (h w) -> b c h w", h=resolution, w=resolution)  # 8 16 64 64
        feature_map = self.dwc(feature_map).mean(1).squeeze()           # 8 64 64
        feature_map = rearrange(feature_map, "b h w -> b (h w)")  # 8 4096
        feature_map = torch.mean(feature_map, dim=0, keepdim=True)   # 1 4096
        feature_map = (feature_map - feature_map.mean()) / feature_map.std()  # 标准化
        feature_map = feature_map.sigmoid()        # sigmoid
        #------------------------------------------------------------------------------------#

        # 添加额外的权重
        attn = attn + feature_map
        attn = torch.softmax(attn, dim=-1)
        # print(attn.shape)
        # visualize_attention_maps(rearrange(attn[0][0], "c (h w) -> c h w", h=64, w=64), 3)
  
        # Get output
        out = attn @ v    

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out, attn


class Attention_i2t_dwc(nn.Module):  # 
    """  
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        add_dwc: bool = True,
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

        if add_dwc:
            kernelsize = 5
            self.dwc = nn.Conv2d(in_channels=self.internal_dim // self.num_heads, out_channels=self.internal_dim // self.num_heads, kernel_size=kernelsize,
                             groups=self.internal_dim // self.num_heads, padding=kernelsize // 2)
            # self.dwc = nn.Conv2d(in_channels=self.internal_dim // self.num_heads, out_channels=self.internal_dim // self.num_heads, kernel_size=kernelsize,
            #                  padding=kernelsize // 2)

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
        attn = torch.softmax(attn, dim=-1)   # b 8 4096 9
        # print(attn.shape)
        # visualize_attention_maps(rearrange(attn[0][0], "(h w) c -> c h w", h=64, w=64), 3)  # 9 64 64
        
        # Get output
        out = attn @ v     # 1 8 4096 16 


        #------------------------ add dwc ----------------------------------------------------#
        b, n_heads, n_tokens, c_per_head = q.shape  # 1 8 4096 16
        resolution = int(n_tokens ** 0.5)
        feature_map = q.reshape(-1, n_tokens, c_per_head).permute(0, 2, 1)  # 8 16 4096
        feature_map = rearrange(feature_map, "b c (h w) -> b c h w", h=resolution, w=resolution) # 8 16 64 64
        feature_map = self.dwc(feature_map)       # 8 16 64 64
        feature_map = rearrange(feature_map, "b c h w -> b (h w) c")
        feature_map = feature_map.reshape(b, n_heads, n_tokens, c_per_head)  # b 8 4096 16
        out = out + feature_map
        #------------------------------------------------------------------------------------#
        

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out, attn


class Focus_Attention(nn.Module):
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

        # self.scale = nn.Parameter(torch.zeros(size=(1, 1, self.internal_dim)))  # added for focus attention

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
        # print(q.shape, k.shape)

        #focus
        focusing_factor = 3
        kernel_function = nn.ReLU()
        # scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        # q = q / scale
        # k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        # print(q.shape, k.shape)
        
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)   # 1 8 4096 16

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        # print(attn.shape)

        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v

        out = self._recombine_heads(out)  # 1 4096 128

        out = self.out_proj(out)

        return out, attn



import matplotlib.pyplot as plt

# 可视化单张注意力图
def visualize_attention_map(attention_weights):
    '''
    args:
        attention_weight: Tensor [h w]
    '''
    ax = plt.gca()
    im = ax.imshow(attention_weights.detach().cpu().numpy())
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
# 可视化多张注意力图
def grid_show(to_shows, cols):
    '''
    args:
        to_shows: 列表，保存要可视化的图；
        cols: int, 列数
    '''
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

# 可视化多张图
def visualize_attention_maps(attention_weights, cols):
    '''
    args:
        attention_weights: tensor, [n h w];
        cols: 列数
    '''
    to_shows = []
    att_map = attention_weights.detach().cpu().numpy()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'token {i}'))

    grid_show(to_shows, cols=cols)



import cv2 
def visualize_attention_map_opencv(attention_weights):
    # 将张量移动到 CPU
    attention_weights_cpu = attention_weights

    # 归一化到 [0, 255] 的范围
    normalized_attention = (attention_weights_cpu - attention_weights_cpu.min()) / (attention_weights_cpu.max() - attention_weights_cpu.min()) * 255

    # 使用 OpenCV 绘制热力图
    heatmap = cv2.applyColorMap(np.uint8(normalized_attention), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (1024, 1024))
    
    # 可视化
    # cv2.imshow('Attention Map', heatmap)
    # cv2.imwrite('work_dirs/sam-l-edge-enhancev4/12.11_correct-results/1.jpg', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


