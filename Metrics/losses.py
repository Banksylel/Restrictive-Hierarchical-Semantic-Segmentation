import torch
import torch.nn as nn
import torch.nn.functional as F
from tree_util import getTreeList
import math
import segmentation_models_pytorch as smp
import os
import torchvision
from Metrics import performance_metrics




# non-hierarchal loss functions

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=0, num_classes=3):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        # self.diceloss = smp.losses.DiceLoss(mode='multiclass', classes=num_classes, log_loss=False, from_logits=True, smooth=self.smooth) # , eps=1e-07
    
    # dice calculation
    def dice_loss_equation(self, inputs, targets, class_weights=None, mask=None, dim_select=0, bypass_clss_loop=False):
        # Calculate intersection and union
        intersection, union = 0.0, 0.0
        if class_weights is None:
            class_weights = 1.0
        # loops class
        if not bypass_clss_loop:
            for clss in range(inputs.shape[0]):
                if mask is not None:
                    input_clss = inputs[clss][mask[clss]]
                    target_clss = targets[clss][mask[clss]]
                intersection += (input_clss * target_clss * class_weights[clss]).sum()
                union += (input_clss * class_weights[clss]).sum() + (target_clss * class_weights[clss]).sum()
        else:
            intersection = (inputs * targets * class_weights).sum(dim_select)
            union = (inputs * class_weights).sum(dim_select) + (targets * class_weights).sum(dim_select)
        # Compute Dice score and loss
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss
    
    # Dice Loss Function
    def DiceLoss(self, inputs, targets, ignore=-1.0, class_weight=None):

        if ignore is not None:
            dice_loss = []
            masks = (targets != ignore)
            # if True in mask:
            for batch in range(targets.shape[0]):
                mask = masks[batch]
                target_masked = targets[batch]
                logits_masked = inputs[batch]
                if class_weight is not None:
                    # reshapes class_weight to match dim 1 of other tensors
                    class_weight_reshaped = torch.tensor(class_weight).unsqueeze(1).to(logits_masked.device)
                dice_loss.append(self.dice_loss_equation(logits_masked, target_masked, class_weight_reshaped, mask, dim_select=1))
            # else:
            #     dice_loss.append(torch.tensor(0.0).to(inputs.device))
            # dice_loss = [torch.nan_to_num(l, nan=1e-6) for l in dice_loss]
            # removes nan values from dice_loss list
            dice_loss = [l for l in dice_loss if not torch.isnan(l)]
            # dice_loss = [torch.nan_to_num(l, nan=1.0) for l in dice_loss]
            return torch.stack(dice_loss).mean() if dice_loss else None
        
        else:
            # Calculate intersection and union
            dice_loss = self.dice_loss_equation(inputs, targets, class_weight, dim_select=2, bypass_clss_loop=True)
            

            return dice_loss


    def forward(self, outs, targets, logits_input=False, class_weight=None):
        if logits_input:
            # Apply softmax to get probabilities
            outs = F.softmax(outs, dim=1)
        outs = outs.contiguous().view(outs.size(0), outs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        
        loss = self.DiceLoss(outs, targets, class_weight=class_weight)
        # loss = self.diceloss(logits, targets)

        return loss



class CrossEntropyLoss(nn.Module):
    def __init__(self, smooth=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.smooth = smooth
        
    def CELoss(self, inputs, targets, ignore = -1, class_weight=None):
        # Apply log softmax to inputs to get log probabilities
        if ignore is not None:
            ce_loss = []
            mask = (targets != ignore)
            for batch in range(targets.shape[0]):
                target_masked = targets[batch]
                logits_masked = inputs[batch]
                if class_weight is not None:
                    # reshapes class_weight to match dim 1 of other tensors
                    class_weight_reshaped = torch.tensor(class_weight).unsqueeze(1).to(logits_masked.device)
                else:
                    class_weight_reshaped = 1.0
                loss = 0.0
                for clss in range(logits_masked.shape[0]):
                    input_clss = logits_masked[clss][mask[batch][clss]]
                    target_clss = target_masked[clss][mask[batch][clss]]
                    loss += -(target_clss * input_clss * class_weight_reshaped[clss]).mean()
                loss = loss / logits_masked.shape[0]
                ce_loss.append(loss)

            ce_loss = [torch.nan_to_num(l, nan=1.0) for l in ce_loss]
            # removes nan values from ce_loss list
            # ce_loss = [l for l in ce_loss if not torch.isnan(l)]
            return torch.stack(ce_loss).mean() if ce_loss else None
        else:
            return -(targets * inputs).sum(1).mean()


    def forward(self, outs, targets, logits_input=False, class_weight=None):
        if logits_input:
            # Apply softmax to get probabilities
            outs = F.log_softmax(outs, dim=1)

        outs = outs.contiguous().view(outs.size(0), outs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        loss = self.CELoss(outs, targets, class_weight=class_weight)

        return loss














# hierarchical consistency loss
def hierarchical_consistency_loss(probs_per_level, levels, parent_of, reduction='mean'):
    """
    Enforces || sum_children(P_c) - P_p ||_1 across parents.
    probs_per_level: list of tensors [B, C_L, H, W], per level (probabilities)
    levels: list[list[str]] names per level, aligned with channels in probs
    parent_of: dict child_name -> parent_name (or None at root)
    """
    if probs_per_level is None or levels is None or parent_of is None:
        # no-op if not provided
        return probs_per_level[0].sum()*0 if probs_per_level else 0.0

    total = 0.0
    count = 0
    for L in range(1, len(levels)):
        P_prev = probs_per_level[L-1] # [B, P, H, W]
        P_cur  = probs_per_level[L] # [B, C, H, W]
        for p_idx, p_name in enumerate(levels[L-1]):
            # indices of children at this level
            ch_idx = [i for i, cname in enumerate(levels[L]) if parent_of.get(cname, None) == p_name]
            if not ch_idx:
                continue
            P_children = P_cur[:, ch_idx, :, :].sum(dim=1, keepdim=True)
            diff = (P_children - P_prev[:, p_idx:p_idx+1, :, :]).abs()
            total += diff.mean() if reduction == 'mean' else diff.sum()
            count += 1
    if count == 0:
        return probs_per_level[0].sum()*0
    return total / count


# def grouped_conditional_kl(z_children_all, probs_prev_level, groups, levels_prev):
#     """
#     Optional stabiliser: per parent group, KL(Q_{c|p} || Uniform).
#     z_children_all: logits for concatenated children at a level [B, sum_g, H, W]
#     probs_prev_level: parent probabilities [B, P_prev, H, W]
#     groups: list of tuples (parent_name, [child_names]) for this level
#     levels_prev: list of parent names (channel order at previous level)
#     """
#     if z_children_all is None or probs_prev_level is None or groups is None:
#         return z_children_all.sum()*0

#     kl = 0.0
#     gcount = 0
#     start = 0
#     eps = 1e-6
#     for (pname, chnames) in groups:
#         g = len(chnames)
#         if g == 0:
#             continue
#         z_g = z_children_all[:, start:start+g, :, :] # [B,g,H,W]
#         # gate children by parent marginal (log-bias trick)
#         p_idx = levels_prev.index(pname)
#         P_p = probs_prev_level[:, p_idx:p_idx+1, :, :]
#         Q = torch.softmax(z_g + torch.log(P_p + eps), dim=1).clamp_min(1e-8)
#         U = torch.full_like(Q, 1.0 / g)
#         kl += (Q * (Q.log() - U.log())).mean()
#         gcount += 1
#         start += g
#     if gcount == 0:
#         return z_children_all.sum()*0
#     return kl / gcount