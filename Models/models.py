from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools


import torch.nn as nn
import torch._utils

from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace


# unet
import copy
from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.vision_transformer import _cfg
# import math



############# Hierarchical Loss #############

# builds hierarchy dependancies (which classes are in which level, 
# which classes are the parent of other classes, and which classes are the children of their parent)
#    Returns:
#      levels: list[list[str]]      # classes per depth, including parents
#      parent_of: dict[str, str|None]
#      children_of: dict[str, list[str]]
def build_hierarchy_indices(hierarchy):
    levels_dict = get_level_classes(hierarchy, inc_parent=True)
    levels = [levels_dict[k] for k in sorted(levels_dict.keys())]

    parent_of = {}
    children_of = {}

    def dfs(d, parent=None):
        for k, v in d.items():
            parent_of[k] = parent
            if isinstance(v, dict) and len(v) > 0:
                children_of[k] = list(v.keys())
                dfs(v, k)
            else:
                children_of.setdefault(k, [])
    dfs(hierarchy, None)
    return levels, parent_of, children_of


# FiLM conditioning
class FiLM(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.cond_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cond_ch, 2*feat_ch)
        )

    def forward(self, feats: torch.Tensor, cond_map: torch.Tensor):
        if cond_map.dim() == 4:
            cond_vec = self.cond_pool(cond_map).flatten(1)
        else:
            cond_vec = cond_map
        gamma_beta = self.mlp(cond_vec)
        C = feats.size(1)
        gamma, beta = gamma_beta[:, :C], gamma_beta[:, C:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        return feats * gamma + beta



# takes the hierarchy structure dictionary and returns the maximum depth and the total number of classes for each depth
def get_level_classes(hierarchy, depth=0, result=None, inc_parent=False):
    if result is None:
        result = {}
    if not isinstance(hierarchy, dict) or not hierarchy:
        return result
    for key, value in hierarchy.items():
        if depth not in result:
            result[depth] = []
        if inc_parent:
            result[depth].append(key)
        else:
            if not value:
                result[depth].append(key)

        if isinstance(value, dict):
            get_level_classes(value, depth + 1, result, inc_parent)
    return result




############### build Unet ###############

# CODE: https://github.com/milesial/Pytorch-UNet?tab=readme-ov-file
# WEIGHTS: https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x





# build UNet model 

class UNet(nn.Module):
    """
    Flat (type==0): identical API, returns [], logits_flat
    Hierarchical (type==1): probabilistic composition with FiLM conditioning.
    """
    def __init__(self, size=620, n_channels=1, hierarchy={}, model_type=0):
        super(UNet, self).__init__()
        self.model_type = model_type
        self.hierarchy = hierarchy

        # Encoder-decoder as before
        self.inc0 = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        if model_type == 0:
            # non hier, output all leaves
            level_classes = [x for x in get_level_classes(hierarchy, inc_parent=False).values()]
            self.out_flat = outconv(64, sum(len(i) for i in level_classes))
        else:
            # hier, build indices and per-level heads
            levels, parent_of, children_of = build_hierarchy_indices(hierarchy)
            self.levels = levels
            self.parent_of = parent_of
            self.children_of = children_of

            # heads: level 0 predicts parents (sigmoid); deeper levels predict concatenated children logits
            self.heads = nn.ModuleList()
            # level 0
            self.heads.append(outconv(64, len(self.levels[0])))
            # levels >0
            self.child_groups = []  # per L>0, list of (parent_name, [child_names])
            for L in range(1, len(self.levels)):
                groups = []
                for p in self.levels[L-1]:
                    ch = self.children_of.get(p, [])
                    if len(ch) > 0:
                        groups.append((p, ch))
                self.child_groups.append(groups)
                total_children = sum(len(ch) for _, ch in groups)
                self.heads.append(outconv(64, total_children if total_children > 0 else 1))

            # FiLM conditioners from prev level probs to decoder features
            self.films = nn.ModuleList([FiLM(feat_ch=64, cond_ch=len(self.levels[L-1]))
                                        for L in range(1, len(self.levels))])

    def _run_unet(self, x):
        # single unet pass, returns the last decoder feature map at 64ch
        x1 = self.inc0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)
        return d  # [B, 64, H, W]

    def forward(self, x, type=0, hierarchy={}, threshold=0.5):
        if self.model_type == 0 or type == 0:
            d = self._run_unet(x)
            logits = self.out_flat(d)
            return [], logits

        probs_per_level = []
        logits_per_level = []

        # Level 0, parents P_p via sigmoid
        d0 = self._run_unet(x)
        z_parent = self.heads[0](d0)
        P_parent = torch.sigmoid(z_parent)
        probs_per_level.append(P_parent)
        logits_per_level.append(z_parent)

        # Levels >0, grouped softmax (Q_{c|p}) and composition P_c = P_p * Q
        eps = 1e-6
        for L in range(1, len(self.levels)):
            # re-encode image, FiLM condition last features with prev level probs
            dL = self._run_unet(x)
            dL = self.films[L-1](dL, probs_per_level[L-1])

            z_children_all = self.heads[L](dL)
            groups = self.child_groups[L-1]
            if len(groups) == 0:
                # no children at this level
                probs_per_level.append(torch.zeros_like(z_children_all))
                logits_per_level.append(z_children_all)
                continue

            Q_list, P_list = [], []
            start = 0
            for pname, chnames in groups:
                g = len(chnames)
                z_g = z_children_all[:, start:start+g, :, :]
                p_idx = self.levels[L-1].index(pname)
                P_p = probs_per_level[L-1][:, p_idx:p_idx+1, :, :]
                # soft gate log(P_p)
                Q_g = torch.softmax(z_g + torch.log(P_p + eps), dim=1)
                P_c = P_p * Q_g
                Q_list.append(Q_g)
                P_list.append(P_c)
                start += g

            P_level = torch.cat(P_list, dim=1)
            probs_per_level.append(P_level)
            logits_per_level.append(z_children_all)

        return probs_per_level, logits_per_level




############### build hrnet ###############
# CODE: https://github.com/HRNet/HRNet-Semantic-Segmentation?tab=readme-ov-file
# WEIGHTS: https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cocostuff_3617_torch04.pth
# build HRNet model



BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}



class HighResolutionNet(nn.Module):
    """
    Hierarchy-aware HRNet with probabilistic composition and FiLM conditioning.
    Flat mode (model_type==0): unchanged API, returns [], logits.
    Hierarchical (model_type!=0): returns [probs_per_level], [logits_per_level].
    """
    def __init__(self, config, hierarchy={}, model_type=0, **kwargs):
        super(HighResolutionNet, self).__init__()
        global ALIGN_CORNERS
        extra = config.MODEL.EXTRA
        ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS

        self.model_type = model_type
        self.hierarchy = hierarchy

        # stem & stages
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]

        # single 3-channel stem (keep original HRNet behaviour)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
        )

        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = int(np.sum(pre_stage_channels))

        self.shared_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        final_k = extra.FINAL_CONV_KERNEL
        final_pad = 1 if final_k == 3 else 0

        # heads
        if self.model_type == 0:
            # non hierarchy, predict all leaves as before
            level_classes = [x for x in get_level_classes(hierarchy, inc_parent=False).values()]
            num_classes_flat = sum(len(i) for i in level_classes)
            self.classifier = nn.Conv2d(last_inp_channels, num_classes_flat, kernel_size=final_k, stride=1, padding=final_pad)
        else:
            # hierarchical, build indices and per-level heads
            levels, parent_of, children_of = build_hierarchy_indices(hierarchy)
            self.levels = levels
            self.parent_of = parent_of
            self.children_of = children_of

            self.classifiers = nn.ModuleList()
            self.classifiers.append(nn.Conv2d(last_inp_channels, len(self.levels[0]), kernel_size=final_k, stride=1, padding=final_pad))
            self.child_groups = []
            for L in range(1, len(self.levels)):
                groups = []
                for p in self.levels[L-1]:
                    ch = self.children_of.get(p, [])
                    if len(ch) > 0:
                        groups.append((p, ch))
                self.child_groups.append(groups)
                total_children = sum(len(ch) for _, ch in groups)
                self.classifiers.append(nn.Conv2d(last_inp_channels, total_children if total_children>0 else 1, kernel_size=final_k, stride=1, padding=final_pad))

            # FiLM conditioners
            self.films = nn.ModuleList([FiLM(feat_ch=last_inp_channels, cond_ch=len(self.levels[L-1]))
                                        for L in range(1, len(self.levels))])

    # backbone passes
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            reset_multi = True if (multi_scale_output or i != num_modules - 1) else False
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi)
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _forward_backbone(self, x):
        # [B, 3, H, W]
        x = self.stem(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # fuse to highest resolution and concatenate
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        outs = [x[0]]
        for b in range(1, len(x)):
            outs.append(F.interpolate(x[b], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS))
        x = torch.cat(outs, 1)
        x = self.shared_head(x)
        return x

    def forward(self, x):
        in_h, in_w = x.shape[-2], x.shape[-1]

        if self.model_type == 0:
            feats = self._forward_backbone(x)
            logits = self.classifier(feats)
            logits = F.interpolate(logits, size=(in_h, in_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            return [], logits

        probs_per_level, logits_per_level = [], []
        eps = 1e-6

        # Level 0, parents (sigmoid)
        feats0 = self._forward_backbone(x)
        z_parent = self.classifiers[0](feats0)
        z_parent = F.interpolate(z_parent, size=(in_h, in_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        P_parent = torch.sigmoid(z_parent)
        probs_per_level.append(P_parent)
        logits_per_level.append(z_parent)

        # Levels >0, FiLM on last feature map, conditional softmax, prob comp
        for L in range(1, len(self.levels)):
            featsL = self._forward_backbone(x)
            featsL = self.films[L-1](featsL, probs_per_level[L-1])
            z_children_all = self.classifiers[L](featsL)
            z_children_all = F.interpolate(z_children_all, size=(in_h, in_w), mode='bilinear', align_corners=ALIGN_CORNERS)

            groups = self.child_groups[L-1]
            if len(groups) == 0:
                probs_per_level.append(torch.zeros_like(z_children_all))
                logits_per_level.append(z_children_all)
                continue

            P_list = []
            start = 0
            for pname, chnames in groups:
                g = len(chnames)
                z_g = z_children_all[:, start:start+g, :, :]
                p_idx = self.levels[L-1].index(pname)
                P_p = probs_per_level[L-1][:, p_idx:p_idx+1, :, :]
                Q_g = torch.softmax(z_g + torch.log(P_p + eps), dim=1)
                P_c = P_p * Q_g
                P_list.append(P_c)
                start += g

            P_level = torch.cat(P_list, dim=1)
            probs_per_level.append(P_level)
            logits_per_level.append(z_children_all)
            # probs_per_level_item = torch.argmax(F.softmax(z_children_all, dim=1), dim=1)
            # probs_per_level.append([F.one_hot(probs_per_level_item, num_classes=probs_per_level_item.shape[1]).permute(0, 3, 1, 2).float()])
        
        return probs_per_level, logits_per_level

    def init_weights(self, pretrained='', device='cpu'):
        checkpoint = torch.load(pretrained, map_location=device)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model_dict = self.state_dict()
        new_checkpoint = {}
        stripped = {}
        for k, v in checkpoint.items():
            kk = k
            for prefix in ['model.', 'module.', 'net.', 'network.']:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            stripped[kk] = v
        for mk, mv in model_dict.items():
            if mk in stripped and stripped[mk].size() == mv.size():
                new_checkpoint[mk] = stripped[mk]
            else:
                for ck, cv in stripped.items():
                    if (mk.endswith(ck) or ck.endswith(mk)) and cv.size() == mv.size():
                        new_checkpoint[mk] = cv
                        break
        loaded = set(new_checkpoint.keys())
        missing = set(model_dict.keys()) - loaded
        print(f"Loaded {len(loaded)} / {len(model_dict)} layers.")
        if missing:
            print(f"Missing {len(missing)} layers (first 10): {list(missing)[:10]}")
        model_dict.update(new_checkpoint)
        self.load_state_dict(model_dict)
        return self


