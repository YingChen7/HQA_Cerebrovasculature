# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:00:17 2021

@author: CY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

import numpy as np


class BaseModel(nn.Module):
    """
    Base model with: a encoder
    """

    def __init__(self, base_encoder, params):
        super(BaseModel, self).__init__()

        # create the encoders
        self.encoder = base_encoder

    def forward(self, x1, x2):
        """
        Input: x1, x2 or x, y_idx
        Output: logits, labels
        """
        raise NotImplementedError
        
class MLP3d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP3d, self).__init__()

        self.linear1 = nn.Conv3d(in_dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 =  nn.Conv3d(inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x
       

def Proj_Head(in_dim=320, inner_dim=640, out_dim=320):
    return MLP3d(in_dim, inner_dim, out_dim)

def Pred_Head(in_dim=320, inner_dim=320, out_dim=320):
    return MLP3d(in_dim, inner_dim, out_dim)

def flip(x, dim=0):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim)-1, -1, -1, dtype = torch.long, device = x.device)
    return x[tuple(indices)]
    
def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5, anatomy=True):
    """ q, k: N * C * H * W * D
        coord_q, coord_k: N * 6 (x_upper_left, y_upper_left, z_upper_left, x_lower_right, y_lower_right, z_lower_right)
    """
    N, C, H, W, D = q.shape
    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)
    
    z_array = torch.arange(0., float(D), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, 1, -1).repeat(1, H, W, 1)
    y_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1, 1).repeat(1, H, 1, D)
    x_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1, 1).repeat(1, 1, W, D)
    
    
    # [bs, 1, 1, 1]
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 0]) / H).view(-1, 1, 1, 1)
    q_bin_width = ((coord_q[:, 4] - coord_q[:, 1]) / W).view(-1, 1, 1, 1)
    q_bin_depth = ((coord_q[:, 5] - coord_q[:, 2]) / D).view(-1, 1, 1, 1)
    
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 0]) / H).view(-1, 1, 1, 1)
    k_bin_width = ((coord_k[:, 4] - coord_k[:, 1]) / W).view(-1, 1, 1, 1)
    k_bin_depth = ((coord_k[:, 5] - coord_k[:, 2]) / D).view(-1, 1, 1, 1)
    
    q_start_x = coord_q[:, 0].view(-1, 1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1, 1)
    q_start_z = coord_q[:, 2].view(-1, 1, 1, 1)
    
    k_start_x = coord_k[:, 0].view(-1, 1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1, 1)
    k_start_z = coord_k[:, 2].view(-1, 1, 1, 1)
    
    q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2 + q_bin_depth**2)
    k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2 + k_bin_depth**2)
    
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)
    
    center_q_x = (x_array + 0.5) * q_bin_height + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_width + q_start_y
    center_q_z = (z_array + 0.5) * q_bin_depth + q_start_z
    
    center_k_x = (x_array + 0.5) * k_bin_height + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_width + k_start_y
    center_k_z = (z_array + 0.5) * k_bin_depth + k_start_z
    
    # [bs, ]
    dist_center = torch.sqrt((center_q_x.view(-1, H * W * D, 1) - center_k_x.view(-1, 1, H * W * D)) ** 2
                             + (center_q_y.view(-1, H * W * D, 1) - center_k_y.view(-1, 1, H * W * D)) ** 2
                             + (center_q_z.view(-1, H *W *D, 1) - center_k_z.view(-1, 1, H *W *D )) **2 ) / max_bin_diag
    
    
    pos_mask = (dist_center < pos_ratio).float().detach()
    
    logit = torch.bmm(q.transpose(1, 2), k)
    
    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1)+ 1e-6)
    
    if anatomy:
        q_rev = flip(q).contiguous()
        center_q_x_rev = flip(center_q_x).contiguous()
        center_q_y_rev = flip(center_q_y).contiguous()
        center_q_z_rev = flip(center_q_z).contiguous()
        dist_center_rev = torch.sqrt((center_q_x_rev.view(-1, H * W * D, 1) - center_k_x.view(-1, 1, H * W * D)) ** 2
                             + (center_q_y_rev.view(-1, H * W * D, 1) - center_k_y.view(-1, 1, H * W * D)) ** 2
                             + (center_q_z_rev.view(-1, H *W *D, 1) - center_k_z.view(-1, 1, H *W *D )) **2 ) / max_bin_diag
        pos_mask_rev =  (dist_center_rev < pos_ratio).float().detach() 
        logit_rev = torch.bmm(q_rev.transpose(1, 2), k)
        loss_rev =  (logit_rev * pos_mask_rev).sum(-1).sum(-1) / (pos_mask_rev.sum(-1).sum(-1)+ 1e-6)
                
    return -2 * loss.mean() - 2 * loss_rev.mean()
    


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input = output
        return input
 
class VessPro(BaseModel):
    def __init__(self, base_encoder, params):
        super(VessPro, self).__init__(base_encoder, params)

        # parse arguments
        self.VessPro_p               = params['VessPro_p']
        self.VessPro_momentum        = params['VessPro-momentum']
        self.VessPro_pos_ratio       = params['VessPro-pos-ratio']
        self.VessPro_clamp_value     = params['VessPro-clamp-value']
        
        self.VessPro_transform_layer = params['VessPro_transform_layer']
        self.VessPro_ins_loss_weight = params['VessPro-ins-loss-weight']
        
        self.encoder = base_encoder     
        self.projector = Proj_Head(in_dim=params['encoder_final_channels'], inner_dim=2*params['encoder_final_channels'])
        
        self.encoder_k = base_encoder
        self.projector_k = Proj_Head(in_dim=params['encoder_final_channels'], inner_dim=2*params['encoder_final_channels'])
        
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        '''    
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        '''
        self.K = int(params['step_size'] * 1. / params['batch_size'] * params['epochs'])
        self.k = int(params['step_size'] * 1. / params['batch_size'] * (params['start_epoch'] - 1))
        
        if self.VessPro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.VessPro_transform_layer == 1:
            self.value_transform = nn.Conv3d(in_channels=params['encoder_final_channels'], out_channels=params['encoder_final_channels'], kernel_size=1, stride=1, padding=0, bias=True)
        elif self.VessPro_transform_layer == 2:
            self.value_transform = MLP3d(in_dim=params['encoder_final_channels'], inner_dim=params['encoder_final_channels'], out_dim=params['encoder_final_channels'])
        else:
            raise NotImplementedError
        
        if self.VessPro_ins_loss_weight > 0.:
            # add instance discrimination
            self.projector_instance = Proj_Head(in_dim=params['encoder_final_channels'], inner_dim=2*params['encoder_final_channels'])
            self.projector_instance_k = Proj_Head(in_dim=params['encoder_final_channels'], inner_dim=2*params['encoder_final_channels'])
            self.predictor = Pred_Head(in_dim=params['encoder_final_channels'], inner_dim=2*params['encoder_final_channels'])
            
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            '''
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
            '''
            self.avgpool = nn.AvgPool3d(7, stride=1)
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.VessPro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1
        
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
          
        if self.VessPro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)
        
               
    def featprop(self, feat):
        N, C, H, W, D = feat.shape
        
        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)
        
        # Similarity calculation 
        feat = F.normalize(feat, dim=1)
        
        feat = feat.view(N, C, -1) # [N, C, H*W*D]
        attention = torch.bmm(feat.transpose(1, 2), feat) # bmm 矩阵乘法
        attention = torch.clamp(attention, min=self.VessPro_clamp_value) # [N, H*W*D, H*W*D]
        if self.VessPro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.VessPro_p
        
        feat = torch.bmm(feat_value, attention.transpose(1, 2))
        
        return feat.view(N, C, H, W, D)    
    
    def regression_loss(self, x, y):
        return -2. * torch.einsum('nc, nc->n', [x, y]).mean()     
        
        
    def forward(self, im_1, im_2, coord1, coord2):
        feat_1 = self.encoder(im_1)  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.featprop(proj_1)
        #pred_1 = self.featprop(feat_1)
        pred_1 = F.normalize(pred_1, dim=1)
        
        feat_2 = self.encoder(im_2)
        proj_2 = self.projector(feat_2)
        pred_2 = self.featprop(proj_2)
        #pred_2 = self.featprop(feat_2)
        pred_2 = F.normalize(pred_2, dim=1)
        
        if self.VessPro_ins_loss_weight > 0.:
            proj_instance_1 = self.projector_instance(feat_1)
            pred_instacne_1 = self.predictor(proj_instance_1)
            pred_instance_1 = F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0), -1), dim=1)

            proj_instance_2 = self.projector_instance(feat_2)
            pred_instance_2 = self.predictor(proj_instance_2)
            pred_instance_2 = F.normalize(self.avgpool(pred_instance_2).view(pred_instance_2.size(0), -1), dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            feat_1_ng = self.encoder_k(im_1)  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)
            #proj_1_ng = F.normalize(feat_1_ng, dim=1)
        
            feat_2_ng = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)
            #proj_2_ng = F.normalize(feat_2_ng, dim=1)
            
            if self.VessPro_ins_loss_weight > 0.:
                proj_instance_1_ng = self.projector_instance_k(feat_1_ng)
                proj_instance_1_ng = F.normalize(self.avgpool(proj_instance_1_ng).view(proj_instance_1_ng.size(0), -1),
                                                 dim=1)

                proj_instance_2_ng = self.projector_instance_k(feat_2_ng)
                proj_instance_2_ng = F.normalize(self.avgpool(proj_instance_2_ng).view(proj_instance_2_ng.size(0), -1),
                                                 dim=1)
        
        loss = regression_loss(pred_1, proj_2_ng, coord1, coord2, self.VessPro_pos_ratio) \
            + regression_loss(pred_2, proj_1_ng, coord2, coord1, self.VessPro_pos_ratio)
            
        if self.VessPro_ins_loss_weight > 0.:
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) + \
                         self.regression_loss(pred_instance_2, proj_instance_1_ng)
            loss = loss + self.VessPro_ins_loss_weight * loss_instance

        return loss

        
        