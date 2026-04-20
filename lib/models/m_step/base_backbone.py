from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.m_step.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg=None, patch_start_index=1):
        # 根据模型进行微调
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER 
        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2) # 位置编码

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2) # 位置编码

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        
        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, track_query=None,token_type="add",mask=None):
        B = x.shape[0]
        x = self.patch_embed(x) # [20, 256, 4, 4, 128]
        
        z = torch.stack(z, dim=1) # 拼接模板
        _, T_z, C_z, H_z, W_z = z.shape #[B,num_t,C,H,W]
        z = z.flatten(0, 1) # [B*num_t,C,H,W]
        # 抽取模板帧的多尺度特征
        z18 = self.patch_embed18(z)
        z16 = self.patch_embed16(z)
        z14 = self.patch_embed14(z)
        # #
        prompt_ms = torch.cat([z18, z16, z14], dim=2).transpose(1, 2)
        prompt_ms = self.MSPG(prompt_ms)
        
        z = self.patch_embed(z) # [60, 64, 4, 4, 128]
        
        if self.add_cls_token:
            # 时序token
            if token_type == "concat":
                new_query = self.cls_token.expand(B, -1, -1)
                query = new_query if track_query is None else torch.cat([new_query, track_query], dim=1)
                query = query + self.cls_pos_embed
            elif token_type == "add":
                query = self.cls_token if track_query is None else track_query + self.cls_token   # self.cls_token is init query
                query = query.expand(B, -1, -1)  # copy B times
                query = query + self.cls_pos_embed
    
        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)

        x = x[..., 0, 0, :] #[B,N,C]
        z = z[..., 0, 0, :] #
        # 经过两个阶段 128*4
        z += self.pos_embed_z # [60, 64, 512]
        x += self.pos_embed_x # [20, 256, 512] 
        _,_,dim = x.shape        
        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous() #[B,N,HW,C]
            z = z.flatten(1, 2)

        lens_x = x.shape[1] # 256
        prompt_ms = prompt_ms.view(B,-1,dim)
        _,pn,_ = prompt_ms.shape
        z = combine_tokens(z, prompt_ms, mode=self.cat_mode) # [8,243,512]
        lens_z = z.shape[1] #
        x = combine_tokens(z, x, mode=self.cat_mode) #[20,448,512] 256+243 = 499
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)  #[B,500,512]
        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x,lens_z,lens_x)
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        
        tq_1,_,p = torch.split(x,[1,lens_z,lens_x],dim=1)
        k = torch.cat([tq_1, prompt_ms], dim=1)
        
        k = self.memory(k)
        tq_2,_ = torch.split(k,[1,pn],dim=1)
        m = tq_1+tq_2
        m = self.sig(m)
        tq = tq_1*(1-m) + m*tq_2
        tq = self.tpe(tq)
        aux_dict = {"attn": None}
        x = self.norm_(x) # [1,681,512]
        return x, tq, aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        if "token_type" in kwargs.keys():
            x, q,aux_dict = self.forward_features(z, x, track_query=kwargs['track_query'], token_type=kwargs['token_type'])
        else:
            x, q, aux_dict = self.forward_features(z, x, track_query=kwargs['track_query'])
        return x, q , aux_dict