import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from networks.bra_block import Block
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from networks.bra_decoder_expandx4 import BasicLayer_up
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from .FCS_attention import DANetHead, MultiSpectralAttentionLayer


class FcsAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(FcsAttention, self).__init__()
        c2wh = dict([(192, 56), (384, 28), (768, 14)])
        # 确保 out_channels 在 c2wh 字典中是有效的键
        if out_channels not in c2wh:
            raise ValueError(f"out_channels value {out_channels} is not supported.")

        self.spatial = DANetHead(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=nn.BatchNorm2d
        )
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            dct_h=c2wh[out_channels],
            dct_w=c2wh[out_channels],
            reduction=reduction,
            freq_sel_method='top16'
        )

    def forward(self, x):
        x = self.frequency_channel(x)
        x = self.spatial(x)
        return x



class PatchExpand(nn.Module):  #上采样长宽各放大2倍，通道数减少2倍
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = x.permute(0,2,3,1)
        x = self.expand(x)
        B, H, W, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):  #最后一步上采样,长宽各放大4倍，通道数
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x

class BRAUnetSystem(nn.Module):
    def __init__(self, img_size=256,depth=[3, 4, 8, 3],depths_decoder=[2,2,2,2], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0.,
                 use_checkpoint_stages=[],
                 norm_layer=nn.LayerNorm,
                 ########
                 n_win=7,
                 kv_downsample_mode='identity',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 final_upsample = "expand_first",
                 mlp_dwconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim[0]  # num_features for consistency with other models
        patches_resolution = [img_size // 4, img_size // 4]
        self.num_layers = len(depth)
        self.patches_resolution = patches_resolution
        self.final_upsample = final_upsample

        self.bn = nn.BatchNorm2d(num_classes)

        # self.SCCSA1 = SCCSA(in_channels=embed_dim[1], out_channels=embed_dim[1])
        # self.SCCSA2 = SCCSA(in_channels=embed_dim[2], out_channels=embed_dim[2])
        # self.SCCSA3 = SCCSA(in_channels=embed_dim[3], out_channels=embed_dim[3])
        # self.SCCSA_pred = SCCSA(in_channels=4 * 9, out_channels=4*9)
        # self.SCCSA_multifusion = SCCSA(in_channels=4*embed_dim[0], out_channels=4*embed_dim[0],reduction_ratio=16)

        # self.DA1 = DANetHead(in_channels=embed_dim[1], out_channels=embed_dim[1], norm_layer=nn.BatchNorm2d)
        # self.DA2 = DANetHead(in_channels=embed_dim[2], out_channels=embed_dim[2], norm_layer=nn.BatchNorm2d)
        # self.DA3 = DANetHead(in_channels=embed_dim[3], out_channels=embed_dim[3], norm_layer=nn.BatchNorm2d)

        self.FCS1 = FcsAttention(in_channels=embed_dim[1], out_channels=embed_dim[1])
        self.FCS2 = FcsAttention(in_channels=embed_dim[2], out_channels=embed_dim[2])
        self.FCS3 = FcsAttention(in_channels=embed_dim[3], out_channels=embed_dim[3])

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=4, mode='nearest'),  # idx0，8倍上采样
            nn.Upsample(scale_factor=2, mode='nearest'),  # idx1，4倍上采样
            nn.Upsample(scale_factor=1, mode='nearest')  # idx2，2倍上采样
        ])

        # 定义卷积层以生成最终的预测
        self.pred_convs = nn.ModuleList([
            nn.Conv2d(in_channels=embed_dim[2], out_channels=num_classes, kernel_size=1),  # 对于最小尺度的预测
            nn.Conv2d(in_channels=embed_dim[1], out_channels=num_classes, kernel_size=1),  # ...
            nn.Conv2d(in_channels=embed_dim[0], out_channels=num_classes, kernel_size=1),  # ...
            nn.Conv2d(in_channels=embed_dim[0], out_channels=num_classes, kernel_size=1)  # 对于最大尺度的预测
        ])

        # 定义最终融合预测的归一化层
        self.pred_norm = nn.BatchNorm2d(num_classes)
        self.pred_conv = nn.Conv2d(in_channels=4*num_classes, out_channels=num_classes, kernel_size=1)

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],
                        input_resolution=(patches_resolution[0] // (2 ** i),
                                          patches_resolution[1] // (2 ** i)),
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*embed_dim[self.num_layers - 1 - i_layer],
                                      embed_dim[self.num_layers - 1 - i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=embed_dim[self.num_layers - 1 - i_layer], dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=embed_dim[self.num_layers - 1 - i_layer],
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths_decoder[i_layer],
                    embed_dim=embed_dim [self.num_layers - 1 - i_layer],
                    num_heads=nheads[(self.num_layers - 1 - i_layer)],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=-1,
                    topks=topks[3 - i_layer],
                    qk_dims=qk_dims[3 - i_layer],
                    n_win=n_win,
                    kv_per_wins=kv_per_wins[3 - i_layer],
                    kv_downsample_kernels=[3 - i_layer],
                    kv_downsample_ratios=[3 - i_layer],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing,
                    soft_routing=soft_routing,
                    pre_norm=pre_norm,
                    mlp_ratios=mlp_ratios[3 - i_layer],
                    mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    qk_scale=qk_scale,
                    before_attn_dwconv=before_attn_dwconv,
                    auto_pad=auto_pad,
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm_up = norm_layer(embed_dim[0])
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up4 = FinalPatchExpand_X4(input_resolution=(img_size // 4, img_size // 4),
                                          dim_scale=4, dim=num_classes)

        self.output = nn.Conv2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)
        self.layers_up2fusion = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_up = PatchExpand(
                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                  patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                dim=embed_dim[self.num_layers - 1 - i_layer], dim_scale=2, norm_layer=norm_layer)
            self.layers_up2fusion.append(layer_up)
        self.FinalConcatDimBack = nn.Linear(4 * embed_dim[0],
                                  embed_dim[0])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def tensor_BLC_2_BLHW(self,x):  # [B,L,C]--→[B,L,H,W]
        B, L, C = x.shape
        x = x.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C)
        x = x.permute(0, 3, 1, 2)
        return x
    def forward_features(self, x):  #对输入x进行三次下采样(downsample_layers)，
        x_downsample = []           #每次下采样后送入transformer模块（stage）,stage输出保存在数组x_downsample中
        for i in range(3):
            x = self.downsample_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = x.flatten(2).transpose(1, 2)
            x = self.stages[i](x)
            x_downsample.append(x)
            B, L, C = x.shape
            x = x.reshape(B,int(math.sqrt(L)),int(math.sqrt(L)),C)
            x = x.permute(0,3,1,2)
        x = self.downsample_layers[3](x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                x_upsample.append(x)
            elif inx == 1:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.tensor_BLC_2_BLHW(x)
                x = self.FCS3(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                x_upsample.append(x)
            elif inx == 2:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.tensor_BLC_2_BLHW(x)
                x = self.FCS2(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                x_upsample.append(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x=self.tensor_BLC_2_BLHW(x)
                # x = self.FCS1(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                x_upsample.append(x)
        return x_upsample

    def features_to_predictions(self, x_upsample):
        multiscale_preds = []
        for idx, x in enumerate(x_upsample):
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

            # 使用预定义的卷积层生成预测
            x = self.pred_convs[idx](x)
            x = F.relu(x)
            multiscale_preds.append(x)

        upsampled_preds = []
        for idx, pred in enumerate(multiscale_preds):
            if idx < 3:  # 对于前三个预测，应用对应的上采样
                upsampled_pred = self.upsamples[idx](pred)
                upsampled_preds.append(upsampled_pred)
            else:  # 最后一个预测不需要上采样
                upsampled_preds.append(pred)

        # 融合所有尺寸的预测结果
        fused_prediction = sum(upsampled_preds)
        # fused_prediction = torch.cat(upsampled_preds, dim=1)
        #
        # fused_prediction = self.SCCSA_pred(fused_prediction)
        # fused_prediction = self.pred_conv(fused_prediction)
        # 应用归一化
        fused_prediction = self.pred_norm(fused_prediction)
        fused_prediction = F.softmax(fused_prediction, dim=1)

        return fused_prediction



    def up_x4(self, x):
        B, C, H, W = x.size()

        # 调整 x 的形状以匹配 FinalPatchExpand_X4 的输入要求
        x = x.view(B, C, H * W).permute(0, 2, 1)  # 转换成 (B, L, C)
        if self.final_upsample == "expand_first":
            x = self.up4(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x_upsample = self.forward_up_features(x, x_downsample)
        # x_upsample = self.custom_fusion_method(x_upsample, x_downsample)
        # x = self.MultiScaleFusion(x_upsample)
        x = self.features_to_predictions(x_upsample)
        x = self.up_x4(x)
        return x
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.stages):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops