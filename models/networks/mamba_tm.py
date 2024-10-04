from __future__ import annotations
from typing import Sequence
import torch.nn as nn
from mamba_ssm import Mamba
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
from .transmorph import (Mlp,
                         PatchMerging,
                         PatchEmbed,
                         SinPositionalEncoding3D,
                         DecoderBlock,
                         Conv3dLReLU,
                         FlowConv)
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from models import FLOW_ESTIMATORS


class MambaMlpBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v2",
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B, n_tokens, C = x.shape
        assert C == self.dim
        x_mamba = self.mamba(self.norm1(x))
        x_mamba = x + self.drop_path(x_mamba)
        x_out = self.mlp(self.norm2(x_mamba))
        x_out = x_mamba + self.drop_path(x_out)
        return x_out


class MambaLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 mlp_ratio=4,
                 drop_path=0.,
                 downsample=None,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 pat_merg_rf=2,
                 ):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            MambaMlpBlock(
                dim,
                mlp_ratio,
                d_state,
                d_conv,
                expand,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ) for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                reduce_factor=pat_merg_rf,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H, W, T: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class MambaEncoder(nn.Module):
    def __init__(
            self,
            pretrain_img_size=224,
            patch_size=4,
            in_chans=2,
            embed_dim=96,
            depths=[2, 2, 4, 2],
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=4,
            drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            spe=False,
            rpe=True,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            pat_merg_rf=2,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # the dropout rate increases from 0 to drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                MambaLayer(
                    dim=int(embed_dim * 2 ** i),
                    depth=depths[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    downsample=PatchMerging if i < self.num_layers - 1 else None,
                    norm_layer=norm_layer,
                    act_layer=nn.GELU,
                    pat_merg_rf=pat_merg_rf,
                )
            )

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # add a norm layer for each output
        for i in out_indices:
            layer = norm_layer(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x: B, C, D, H, W
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed,
                                               size=(Wh, Ww, Wt),
                                               mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MambaEncoder, self).train(mode)
        self._freeze_stages()


@FLOW_ESTIMATORS.register_module()
class Mamba_TM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.encoder = MambaEncoder(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )

        self.up0 = DecoderBlock(embed_dim * 8,
                                embed_dim * 4,
                                skip_channels=embed_dim *
                                              4 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 4,
                                embed_dim * 2,
                                skip_channels=embed_dim *
                                              2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2,
                                embed_dim,
                                skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim,
                                config.reg_head_chan,
                                skip_channels=embed_dim //
                                              2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        # self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dLReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        # self.c2 = Conv3dReLU(2,
        #                      config.reg_head_chan,
        #                      3,
        #                      1,
        #                      use_batchnorm=False)
        self.flow_conv = FlowConv(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """"
        Args:
            source (torch.tensor): [BCHWD]
            target (torch.tensor): [BCHWD]
        Return:
            flow (torch.tensor): [B3, Hf, Wf, Df]
        """
        x = torch.cat([source, target], dim=1)
        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out_feats = self.encoder(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        flow = self.flow_conv(x)
        return flow
