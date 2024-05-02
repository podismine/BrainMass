# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import partial
import numpy as np

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch.nn import TransformerEncoderLayer


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
class FT(nn.Module):
    def __init__(self,feature_dim,depth,heads,dim_feedforward):
        super().__init__()
        self.encoder = BNTF(feature_dim,depth,heads,dim_feedforward)
        self.g2 = nn.Sequential(
            nn.Linear(8 * 100, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,2)
            )
    def forward(self,img):
        bz, _, _, = img.shape

        for atten in self.encoder.attention_list:
            img = atten(img)

        node_feature = self.encoder.dim_reduction(img)
        node_feature = node_feature.reshape((bz, -1))
        node_feature = F.leaky_relu(node_feature)
        node_feature = self.g2(node_feature)
        return node_feature

from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


# class InterpretableTransformerEncoder(TransformerEncoderLayer):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
#                  layer_norm_eps=1e-5, batch_first=False, norm_first=False,
#                  device=None, dtype=None) -> None:
#         super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
#                          layer_norm_eps, batch_first, norm_first, device, dtype)
#         self.attention_weights: Optional[Tensor] = None

#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x, weights = self.self_attn(x, x, x,
#                                     attn_mask=attn_mask,
#                                     key_padding_mask=key_padding_mask,
#                                     need_weights=True,
#                                     average_attn_weights=True)
#         self.attention_weights = weights
#         return self.dropout1(x)

#     def get_attention_weights(self) -> Optional[Tensor]:
#         return self.attention_weights

import torch
from torch.nn import TransformerEncoderLayer

class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(InterpretableTransformerEncoder, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = super().forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        if hasattr(self.self_attn, 'get_attention_map'):
            attn_map = self.self_attn.get_attention_map()
        else:
            attn_map = None
        
        return output, attn_map

class ExplainableBNTF(nn.Module):
    def __init__(self,feature_dim,depth,heads,dim_feedforward):
        super().__init__()
        self.num_patches = 100

        self.attention_list = nn.ModuleList()
        self.node_num = 100
        for _ in range(int(depth)):
            self.attention_list.append(
                InterpretableTransformerEncoder(d_model=self.node_num, nhead=int(heads), dim_feedforward=dim_feedforward, 
                                        batch_first=True)
            )
            # head=10
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.node_num, 8),
            nn.LeakyReLU()
        )

        final_dim = 8 * self.node_num

        self.g = MLPHead(final_dim, final_dim * 2, feature_dim)
        
    def forward(self,img,forward_with_mlp=True):
        bz, _, _, = img.shape
        weights = []
        for atten in self.attention_list:
            img = atten(img)
            atten_weights = atten.get_attention_weights()
            weights.append(atten_weights)
        if forward_with_mlp is not True:
            return img
        node_feature = self.dim_reduction(img)
        node_feature = node_feature.reshape((bz, -1))
        node_feature = self.g(node_feature)
        return node_feature, weights

class BNTF(nn.Module):
    def __init__(self,feature_dim,depth,heads,dim_feedforward):
        super().__init__()
        self.num_patches = 100

        self.attention_list = nn.ModuleList()
        self.node_num = 100
        #for _ in range(12):
        for _ in range(int(depth)):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=self.node_num, nhead=int(heads), dim_feedforward=dim_feedforward, 
                                        batch_first=True)
            )
            # head=10
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.node_num, 8),
            nn.LeakyReLU()
        )

        final_dim = 8 * self.node_num

        #self.g = nn.Sequential(nn.Linear(final_dim, feature_dim),nn.BatchNorm1d(feature_dim))
        #self.g = nn.Sequential(nn.Linear(final_dim, feature_dim))
        self.g = MLPHead(final_dim, final_dim * 2, feature_dim)
        
    def forward(self,img,forward_with_mlp=True):
        bz, _, _, = img.shape

        for atten in self.attention_list:
            img = atten(img)
        if forward_with_mlp is not True:
            return img
        node_feature = self.dim_reduction(img)
        node_feature = node_feature.reshape((bz, -1))
        node_feature = self.g(node_feature)
        return node_feature


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        feature_dim = 256
        self.online_network = BNTF(feature_dim)
        self.target_network = BNTF(feature_dim)
        # self.softmax = nn.Softmax(dim=-1)
        # self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self,img1,img2,ftype='train'):
        if 'train' in ftype:
            batch_size,M,C = img1.shape
            pred_1 = self.fc(self.online_network(img1))
            pred_2 = self.fc(self.online_network(img2))

            out_1 = F.normalize(out_1, dim=-1)
            out_2 = F.normalize(out_2, dim=-1)

            temperature=0.5
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix,device=img1.device) - torch.eye(2 * batch_size, device=img1.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            return loss
        elif 'test' in ftype:
            fea = self.BNTF(img1)
            #print(fea)
            fea = F.normalize(fea,dim=-1)
            #print(fea)
            out = self.fc(fea)
            #print(out)
            #exit()
            return out
        else:
            print("ftype error")
            exit()


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == patch_size ** 1
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        #self.encoder_to_decoder = nn.Linear(encoder_embed_dim, 2)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, img1,img2,ftype='train'):
        
        x_vis = self.encoder(img1,img2,ftype) # [B, N_vis, C_e]
        return x_vis

@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=100,
        patch_size=100,
        encoder_embed_dim=100,
        encoder_depth=4,
        encoder_num_heads=4,
        encoder_num_classes=0,
        decoder_num_classes=100,
        decoder_embed_dim=100,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 

@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model