import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from lib.models.m_step.base_backbone import BaseBackbone
from einops import rearrange
from mamba_ssm import Mamba


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0,attn_type="concat"):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_type = attn_type

    def forward(self, x, lens_z=None , lens_x=None, return_attention=False):
        if self.attn_type=='concat':
            #print("concat")
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # 2024/7/19 添加
            attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.attn_type == 'separate':
            # 加入时序token
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  
            # make torchscript happy (cannot use tensor as tuple)
            q_track, q_t, q_s = torch.split(q, [1, lens_z, lens_x], dim=2)
            t_track, k_t, k_s = torch.split(k, [1, lens_z, lens_x], dim=2)
            v_track, v_t, v_s = torch.split(v, [1, lens_z, lens_x], dim=2)
            # template attention 模版做自注意力
            attn = (q_t @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')
            # search region attention 
            k_ts = torch.cat([k_t, k_s], dim=2) # 可以尝试使用三个特征cat
            v_ts = torch.cat([v_t, v_s], dim=2)
            attn = (q_s @ k_ts.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_s = rearrange(attn @ v_ts, 'b h t d -> b t (h d)')
            # track_query attention
            attn = (q_track @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_track = rearrange(attn @ v, 'b h t d -> b t (h d)')
            
            x = torch.cat([x_track, x_t, x_s], dim=1)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.attn_type == 'mix':
            # print('mix')
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # 
            q_track,q_t = torch.split(q, [1, N-1], dim=2)
            k_track,k_t = torch.split(k, [1, N-1], dim=2)
            v_track,v_t = torch.split(v, [1, N-1], dim=2)
            # template attention 模版做自注意力
            attn = (q_t @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')
            # track_query attention
            attn = (q_track @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_track = rearrange(attn @ v, 'b h t d -> b t (h d)')
            x = torch.cat([x_track, x_t], dim=1)
            x = self.proj(x)
            x = self.proj_drop(x)
        if return_attention:
                return x, attn
        return x
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class MambaBlock(nn.Module):
    def __init__(self, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_type="concat" ,attn_drop=0., drop_path=0., rpe=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,init_values=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim) 
        
        # 添加Mamba
        self.mamba = MambaLayer(
                dim,d_state=64,
                d_conv=4, 
                expand=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_type="concat" ,attn_drop=0., drop_path=0., rpe=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,init_values=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        
        self.attn = (
           Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_type=attn_type
            )
            if with_attn
            else None
        )
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, lens_z=None,lens_x=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), lens_z=None,lens_x=None))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
 

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1, 
            patches_resolution[0], self.inner_patches, 
            patches_resolution[1], self.inner_patches, 
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
    
    def forward(self, x):
        # 可以考虑改进
        x0 = x[..., 0::2, 0::2, :] 
        x1 = x[..., 1::2, 0::2, :] 
        x2 = x[..., 0::2, 1::2, :] 
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)# 降维
        return x
    
class PatchEmbed1(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = x.transpose(1,2).view(B, C, H, W)
        return x
    

class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            if (i == 0):
                x_flatten = tensor.view(num, c, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num,c,-1)), dim=2)
        return x_flatten
    
    
    
class MSTE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mid_chans=64,in_chans=3,embed_dim=768, norm_layer=None, flatten=True,rate=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size// patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj1 = nn.Conv2d(in_chans, mid_chans, kernel_size=3, stride=1,dilation=rate)
        self.proj2 = nn.Conv2d(mid_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spp = SPPLayer(3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.spp(x)
        x = self.norm(x)
        return x

class HiViT(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 add_cls_token=False, attn_type="concat",
                 **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.ape = ape # 绝对位置编码
        self.rpe = rpe # 相对位置编码
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]
        end_dim = embed_dim
        embed_dim = embed_dim // 2 ** (self.num_layers - 1) ##这里改变embed_dim 512/4=128
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        
        self.num_tokens = 2 
        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, end_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, end_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] += Hp - 1 
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))  # stochastic depth decay rule

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for i in range(stage_depth):
                self.blocks.append(
                   Block(
                        dim=embed_dim,
                        num_heads=nhead,
                        mlp_ratio=ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=next(dpr),  # dpr[i]
                        norm_layer=norm_layer,
                        attn_type=attn_type
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2
        self.norm_ = norm_layer(embed_dim)
        # MSTE
        self.patch_embed14 = MSTE(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=end_dim,rate=1)

        self.patch_embed16 = MSTE(
            img_size=img_size, patch_size=patch_size+2, in_chans=in_chans, embed_dim=end_dim,rate=2)
        # 调整了rate

        self.patch_embed18 = MSTE(
            img_size=img_size, patch_size=patch_size+4, in_chans=in_chans, embed_dim=end_dim,rate=5)
        
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.MSPG = nn.Linear(end_dim, end_dim,bias=False)
        # 2024/7/7 使用Mamba来做时序token增强
        self.mamba = MambaBlock(dim = embed_dim)
        # 只使用MS 这里使用的是mix
        self.memory = Block(dim=embed_dim,num_heads=nhead,mlp_ratio=ratio,qkv_bias=qkv_bias,
                            drop=drop_rate,attn_drop=attn_drop_rate,
                            norm_layer=norm_layer,
                            attn_type='concat' # adapter
                    )
        # 2024/7/16
        self.sig = nn.Sigmoid()
        self.memory_norm = norm_layer(embed_dim)
        self.tpe = nn.Linear(end_dim, end_dim,bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

def _create_vision_transformer(pretrained=False, default_cfg=None, **kwargs):
    
    model = HiViT(**kwargs)
    #print(model.blocks)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(missing_keys, unexpected_keys)
        print('Load pretrained model from: ' + pretrained)

    return model


def hivit_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
    return model

def hivit_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dim=384, depths=[2, 2, 20], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., 
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs
    )
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    
    return model