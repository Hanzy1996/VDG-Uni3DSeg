"""
Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from mmdet3d.registry import MODELS

from einops import rearrange

@MODELS.register_module()
class Contrast_Criteria:
    def __init__(self, 
                 loss_weight,
                 ce_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True)):

        self.ce_loss = MODELS.build(ce_loss)
        self.loss_weight = loss_weight

    def get_layer_loss(self, layer, aux_outputs, gt_sem):
        sem_attns = aux_outputs
        loss = []
        for pred_mask, gt_masks in zip(sem_attns, gt_sem):
            gt_mask = gt_masks.sp_masks.float().argmax(0) # shape 24590
            repeat_gt_mask = gt_mask.repeat(pred_mask.shape[1], 1).flatten() # shape 5, 24590
            pred_mask_reshape = rearrange(pred_mask, 'c n p -> (n p) c')
            loss.append(self.ce_loss(pred_mask_reshape, repeat_gt_mask))
        contrast_loss = self.loss_weight * torch.mean(torch.stack(loss))
        return {f'layer_{layer}_contrast': contrast_loss}
    
    def __call__(self, sem_attns, gt_sem, sem_aux_outputs=None):
        
        losses = []
        for pred_mask, gt_masks in zip(sem_attns, gt_sem):
            gt_mask = gt_masks.sp_masks.float().argmax(0) # shape 24590
            repeat_gt_mask = gt_mask.repeat(pred_mask.shape[1], 1).flatten() # shape 5, 24590
            pred_mask_reshape = rearrange(pred_mask, 'c n p -> (n p) c')
            loss = self.ce_loss(pred_mask_reshape, repeat_gt_mask)
            losses.append(loss)
        contrast_loss = self.loss_weight * torch.mean(torch.stack(losses))
        loss = {'contrast_loss': contrast_loss}


        if sem_aux_outputs is not None:
            for i, aux_outputs in enumerate(sem_aux_outputs):
                loss_i = self.get_layer_loss(i, aux_outputs, gt_sem)
                loss.update(loss_i)
        return loss
    


@MODELS.register_module()
class Contrast_Criteria_v2:
    def __init__(self, 
                 loss_weight,
                 num_classes,
                 des_proto,
                 img_proto,
                 temp=1.0,
                 ce_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True)):

        self.ce_loss = MODELS.build(ce_loss)
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.des_proto = des_proto
        self.img_proto = img_proto
        self.temp = temp

    def get_contras_loss(self, pred_mask, gt_sem):

        # n: point number, c: class number, p: prototype number
        mask = rearrange(gt_sem, 'n c p -> (c n) p')

        pred_mask = rearrange(pred_mask / self.temp, 'c n p -> (c n) p')

        logits_max, _ = torch.max(pred_mask, dim=-1, keepdim=True)

        logits = pred_mask - logits_max

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits.sum(-1, keepdim=True))

        single_samples = (mask.sum(-1, keepdims=True) == 0).float()

        mean_log_prob_pos = (mask * log_prob).sum(-1, keepdims=True) /(mask.sum(-1, keepdims=True)+single_samples)

        contrast_loss = - mean_log_prob_pos  # mean_log_prob_pos*(1-single_samples)

        contrast_loss = contrast_loss.mean()

        return contrast_loss

    def get_layer_loss(self, layer, aux_outputs, gt_sem):
        sem_attns = aux_outputs
        loss = []
        for pred_mask, gt_masks in zip(sem_attns, gt_sem):
            gt_mask = gt_masks.sp_masks.float() # shape 24590
            repeat_gt = gt_mask.unsqueeze(0).repeat(self.des_proto+self.img_proto, 1, 1)
            contrast_loss = self.get_contras_loss(pred_mask, repeat_gt)
            loss.append(contrast_loss)
        contrast_loss = self.loss_weight * torch.mean(torch.stack(loss))
        return {f'layer_{layer}_contrast': contrast_loss}
    
    def __call__(self, sem_attns, gt_sem, sem_aux_outputs=None):
        
        losses = []
        for pred_mask, gt_masks in zip(sem_attns, gt_sem):
            gt_mask = gt_masks.sp_masks.float() # shape 24590
            repeat_gt = gt_mask.unsqueeze(0).repeat(self.des_proto+self.img_proto, 1, 1)
            contrast_loss = self.get_contras_loss(pred_mask, repeat_gt)
            losses.append(contrast_loss)
        contrast_loss = self.loss_weight * torch.mean(torch.stack(losses))
        loss = {'contrast_loss': contrast_loss}


        if sem_aux_outputs is not None:
            for i, aux_outputs in enumerate(sem_aux_outputs):
                loss_i = self.get_layer_loss(i, aux_outputs, gt_sem)
                loss.update(loss_i)
        return loss
    