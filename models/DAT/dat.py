# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from .dat_blocks import *

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        # 6
        self.depths = depths
        # dim_in 128
        # dim_embed 320
        # heads=5 hc = 64
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        # [2*6=12]
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        # 未采用dwc_mlps,使用最基本的mlp,expansion是指中间特征层维度增加四倍,然后再恢复原维度
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx, sr_ratio, drop_path_rate[i])
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, feat_tgt, feat_ref):
        positions = []
        references = []
        for d in range(self.depths):
            feat_tgt0 = feat_tgt
            feat_tgt = self.layer_norms[2 * d](feat_tgt)
            feat_ref = self.layer_norms[2 * d](feat_ref)
            feat_tgt, pos, ref = self.attns[d](feat_tgt,feat_ref)
            feat_tgt = self.drop_path[d](feat_tgt) + feat_tgt0
            feat_tgt0 = feat_tgt
            feat_tgt = self.mlps[d](self.layer_norms[2 * d + 1](feat_tgt))
            feat_tgt = self.drop_path[d](feat_tgt) + feat_tgt0
            positions.append(pos)
            references.append(ref)

        return feat_tgt, positions, references

class DAT(nn.Module):
    arch_settings = {
        'dat_tiny': {
            'img_size': [32,16],
            'num_classes': 19,
            'expansion': 4,
            'dim_stem': 128,
            'dims': [320, 512],
            'depths': [2, 2],
            'stage_spec': [['L','D'], ['L','D']],
            # heads必须是groups的倍数
            'heads': [4, 8],
            'window_sizes': [4, 4],
            'groups': [4, 8],
            'use_pes': [False, False],
            'dwc_pes': [False, False],
            'strides': [1, 1],
            'sr_ratios': [1, 1],
            'offset_range_factor': [2, 2],
            'no_offs': [False, False],
            'fixed_pes': [False, False],
            'use_dwc_mlps': [True, True],
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.2,
        },
    }
    def __init__(self,model_type = 'dat_tiny', ns_per_pts=[4, 4],**kwargs):
        super().__init__()
        self.img_size = self.arch_settings[model_type]['img_size']
        self.dim_stem = self.arch_settings[model_type]['dim_stem']
        self.drop_path_rate = self.arch_settings[model_type]['drop_path_rate']
        self.depths = self.arch_settings[model_type]['depths']
        self.dims = self.arch_settings[model_type]['dims']
        self.window_sizes = self.arch_settings[model_type]['window_sizes']
        self.stage_spec = self.arch_settings[model_type]['stage_spec']
        self.groups = self.arch_settings[model_type]['groups']
        self.use_pes = self.arch_settings[model_type]['use_pes']
        self.sr_ratios = self.arch_settings[model_type]['sr_ratios']
        self.heads = self.arch_settings[model_type]['heads']
        self.strides = self.arch_settings[model_type]['strides']
        self.offset_range_factor = self.arch_settings[model_type]['offset_range_factor']
        self.dwc_pes = self.arch_settings[model_type]['dwc_pes']
        self.no_offs = self.arch_settings[model_type]['no_offs']
        self.fixed_pes = self.arch_settings[model_type]['fixed_pes']
        self.attn_drop_rate = self.arch_settings[model_type]['attn_drop_rate']
        self.drop_rate = self.arch_settings[model_type]['drop_rate']
        self.expansion = self.arch_settings[model_type]['expansion']
        self.use_dwc_mlps = self.arch_settings[model_type]['use_dwc_mlps']
        self.num_classes = self.arch_settings[model_type]['num_classes']


        # drop_path_rate [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171, 0.2] 8个Transformerblock的数据
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.stages = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        for i in range(2):
            # i=0,dim1=128,dim2=320;i=1,dim1=320,dim1=512
            dim1 = self.dim_stem if i == 0 else self.dims[i - 1]
            dim2 = self.dims[i]
            # Overlapped patch embedding
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dim1, dim2, 3, 2, 1),
                    LayerNormProxy(dim2)
                )
            )
            self.stages.append(
                TransformerStage(self.img_size[i], self.window_sizes[i], ns_per_pts[i],
                dim2, dim2, self.depths[i], self.stage_spec[i], self.groups[i], self.use_pes[i], 
                self.sr_ratios[i], self.heads[i], self.strides[i], 
                self.offset_range_factor[i], i,
                self.dwc_pes[i], self.no_offs[i], self.fixed_pes[i],
                self.attn_drop_rate, self.drop_rate, self.expansion, self.drop_rate, 
                dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])],
                self.use_dwc_mlps[i])
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize the model weights
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def forward(self, feats_tgt, feats_ref):
        out = []
        positions = []
        references = []
        for i in range(1,3):
            feat_tgt = feats_tgt[i]
            feat_ref = feats_ref[i]
            feat_tgt = self.down_projs[i-1](feat_tgt)
            feat_ref = self.down_projs[i-1](feat_ref)
            x, pos, ref = self.stages[i-1](feat_tgt, feat_ref)
            out.append(x)
            positions.append(pos)
            references.append(ref)
        # out:[(2,320,32,32),[2,512,16,16]]

        # Layer Norm
        feats = feats_tgt[:2]+out
        feats[2]+= feats_tgt[2]
        feats[3]+= feats_tgt[3]
        return feats, positions, references
