import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from networks.bra_block import Block
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from networks.bra_decoder_expandx4 import BasicLayer_up
import torch.nn.functional as F


class SCCSA(nn.Module):  #输入张量x（应该是concat融合过后的），进行通道注意力和空间注意力处理，输出原维度的张量x
    def __init__(self, in_channels, out_channels, rate=4):
        super(SCCSA, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


# class LayerNorm1(nn.Module):
#     r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x


class LayerNorm1(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, num_classes, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, kernel_size=1)
        group_size = dim_xl // len(d_list)  # 计算分组大小

        # 分组卷积模块
        self.groups = nn.ModuleList()
        for d in d_list:
            self.groups.append(nn.Sequential(
                LayerNorm1(num_features=group_size + num_classes),
                nn.Conv2d(group_size + num_classes, group_size + num_classes, kernel_size=k_size,
                          padding=k_size // 2 + (d - 1) // 2, dilation=d, groups=1)
            ))

        # 最后的尾部卷积将所有分组的输出特征图整合在一起
        total_channels_after_concat = 4 * (group_size + num_classes) + dim_xl
        self.tail_conv = nn.Conv2d(total_channels_after_concat, dim_xl, kernel_size=1)

    def forward(self, xh, xl, mask):
        # 对xh执行1x1卷积和上采样来匹配xl的尺寸
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=xl.shape[2:], mode='bilinear', align_corners=True)
        # 对mask上采样来匹配xl的尺寸
        mask = F.interpolate(mask, size=xl.shape[2:], mode='bilinear', align_corners=True)
        xh_chunks = torch.chunk(xh, chunks=len(self.groups), dim=1)

        group_outputs = []
        for group, xh_chunk in zip(self.groups, xh_chunks):
            group_output = group(torch.cat((xh_chunk, mask), dim=1))
            # 如果group_output的尺寸与xl不匹配，执行上采样
            if group_output.size(2) != xl.size(2) or group_output.size(3) != xl.size(3):
                group_output = F.interpolate(group_output, size=xl.shape[2:], mode='bilinear', align_corners=True)
            group_outputs.append(group_output)

        # 拼接经过尺寸调整的group_outputs和xl
        aggregated_output = torch.cat((*group_outputs, xl), dim=1)
        # 尾部1x1卷积
        output = self.tail_conv(aggregated_output)
        return output


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

        self.GAB3 = group_aggregation_bridge(dim_xh=embed_dim[3], dim_xl=embed_dim[2], num_classes=num_classes)
        self.GAB2 = group_aggregation_bridge(dim_xh=embed_dim[2], dim_xl=embed_dim[1], num_classes=num_classes)
        self.GAB1 = group_aggregation_bridge(dim_xh=embed_dim[1], dim_xl=embed_dim[0], num_classes=num_classes)

        self.SCCSA1 = SCCSA(in_channels=embed_dim[1], out_channels=embed_dim[1])
        self.SCCSA2 = SCCSA(in_channels=embed_dim[2], out_channels=embed_dim[2])
        self.SCCSA3 = SCCSA(in_channels=embed_dim[3], out_channels=embed_dim[3])
        self.SCCSA_pred = SCCSA(in_channels=4 * 9, out_channels=4*9)
        # self.SCCSA_multifusion = SCCSA(in_channels=4*embed_dim[0], out_channels=4*embed_dim[0],reduction_ratio=16)

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=8, mode='nearest'),  # idx0，8倍上采样
            nn.Upsample(scale_factor=4, mode='nearest'),  # idx1，4倍上采样
            nn.Upsample(scale_factor=2, mode='nearest')  # idx2，2倍上采样
        ])

        # 定义卷积层以生成最终的预测
        self.pred_convs = nn.ModuleList([
            nn.Conv2d(in_channels=embed_dim[3], out_channels=num_classes, kernel_size=1),  # 对于最小尺度的预测
            nn.Conv2d(in_channels=embed_dim[2], out_channels=num_classes, kernel_size=1),  # ...
            nn.Conv2d(in_channels=embed_dim[1], out_channels=num_classes, kernel_size=1),  # ...
            nn.Conv2d(in_channels=embed_dim[0], out_channels=num_classes, kernel_size=1)  # 对于最大尺度的预测
        ])

        # 定义最终融合预测的归一化层
        self.pred_norm = nn.BatchNorm2d(num_classes)
        self.add_norm1 = nn.BatchNorm2d(embed_dim[2])
        self.add_norm2 = nn.BatchNorm2d(embed_dim[1])
        self.add_norm3 = nn.BatchNorm2d(embed_dim[0])
        self.pred_conv = nn.Conv2d(in_channels=36, out_channels=9, kernel_size=1)

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
                                          dim_scale=4, dim=9)

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
        for i in range(4):
            x = self.downsample_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = x.flatten(2).transpose(1, 2)
            x = self.stages[i](x)
            x_downsample.append(x)
            B, L, C = x.shape
            x = x.reshape(B,int(math.sqrt(L)),int(math.sqrt(L)),C)
            x = x.permute(0,3,1,2)
        # x = self.downsample_layers[3](x)
        # print(x.shape)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        x_upsample = []
        x_layerup = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = x.flatten(2).transpose(1, 2)
                x_upsample.append(x)
                x = self.tensor_BLC_2_BLHW(x)
                mask1 = self.pred_convs[0](x)
                mask1 = F.relu(mask1)
                x_temp = x
                x_layerup.append(x_temp)
                # print(x_layerup[0].shape)
                x_layerup[0] = layer_up(x_layerup[0])
                # print(x_layerup[0].shape)
            elif inx == 1:
                x_downsample[2] = self.tensor_BLC_2_BLHW(x_downsample[2])
                # print(x.shape)
                # print(x_downsample[2].shape)
                # print(mask1.shape)
                x = self.GAB3(x, x_downsample[2], mask=mask1)
                x_layerup[0] = self.tensor_BLC_2_BLHW(x_layerup[0])
                x = sum(x, x_layerup[0])
                x = self.add_norm1(x)
                # print(x.shape)
                mask2 = self.pred_convs[1](x)
                mask2 = F.relu(mask2)
                x_temp1 = x.flatten(2).transpose(1, 2)
                x_layerup.append(x_temp1)
                x_upsample.append(x_layerup[1])
                # x_layerup[1] = self.concat_back_dim[inx](x_layerup[1])
                # print(x_layerup[1].shape)
                x_layerup[1] = layer_up(x_layerup[1])
            elif inx == 2:
                x_downsample[1] = self.tensor_BLC_2_BLHW(x_downsample[1])
                x = self.GAB2(x, x_downsample[1], mask=mask2)
                x_layerup[1] = self.tensor_BLC_2_BLHW(x_layerup[1])
                x = sum(x, x_layerup[1])
                x = self.add_norm2(x)
                # print(x.shape)
                mask3 = self.pred_convs[2](x)
                mask3 = F.relu(mask3)
                x_temp2 = x.flatten(2).transpose(1, 2)
                x_layerup.append(x_temp2)
                x_upsample.append(x_layerup[2])
                # x_layerup[1] = self.concat_back_dim[inx](x_layerup[1])
                # print(x_layerup[1].shape)
                x_layerup[2] = layer_up(x_layerup[2])
            else:
                x_downsample[0] = self.tensor_BLC_2_BLHW(x_downsample[0])
                x = self.GAB1(x, x_downsample[0], mask=mask3)
                x_layerup[2] = self.tensor_BLC_2_BLHW(x_layerup[2])
                x = sum(x, x_layerup[2])
                x = self.add_norm3(x)
                x_layerup[2] = x.flatten(2).transpose(1, 2)
                x_upsample.append(x_layerup[2])
                # print(x_layerup[2].shape)
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