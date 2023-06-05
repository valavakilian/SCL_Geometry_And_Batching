"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required, was {}'.format(features.shape))
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features, was {} vs. {}'.format(labels.shape, features.shape))
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() + 1e-12 # added 1e-12 for stability 

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-12 # added 1e-12 for stability 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        if mask.sum(1).all() != 0:
          mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        else:
          #print('FLAG FLAG FLAG')
          idx = (mask.sum(1) == 0).nonzero(as_tuple=True)[0]
          masksum = mask.sum(1)
          #print(masksum.requires_grad)
          #with torch.no_grad():
          masksum[idx] = 1  
          #masksum[idx].detach()
          
          mean_log_prob_pos = (mask * log_prob).sum(1) / masksum
          mean_log_prob_pos[idx] = 0 

        
        #mean_log_prob_pos.register_hook(lambda t: print(f'hook mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) :\n {t}'))
        # loss
        #exp_logits.register_hook(lambda grad: print("max: {}, min {}".format(torch.max(grad).item(), torch.min(grad).item())))
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class WeightedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, num_classes =10, device = None):
        super(WeightedSupConLoss, self).__init__()
        self.temperature = temperature # assume temperature is a 1 x k list of the temperatures for each class 
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.num_classes = num_classes 
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required, was {}'.format(features.shape))
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features, was {} vs. {}'.format(labels.shape, features.shape))
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        
        temp_per_anchor = torch.from_numpy(np.array([np.sqrt(self.temperature[int(i)]) for i in labels])).reshape(-1,1).to(self.device)
        anchor_dot_contrast = torch.matmul(torch.div(anchor_feature,temp_per_anchor.repeat(1,anchor_feature.shape[1])).float(), torch.div(contrast_feature.T,temp_per_anchor.repeat(1,anchor_feature.shape[1]).T).float())
        

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() + 1e-12 # added 1e-12 for stability 

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-12 # added 1e-12 for stability 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        if mask.sum(1).all() != 0:
          mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        else:
          #print('FLAG FLAG FLAG')
          idx = (mask.sum(1) == 0).nonzero(as_tuple=True)[0]
          masksum = mask.sum(1)
          #print(masksum.requires_grad)
          #with torch.no_grad():
          masksum[idx] = 1  
          #masksum[idx].detach()
          
          mean_log_prob_pos = (mask * log_prob).sum(1) / masksum
          mean_log_prob_pos[idx] = 0 

        
        #mean_log_prob_pos.register_hook(lambda t: print(f'hook mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) :\n {t}'))
        # loss
        #exp_logits.register_hook(lambda grad: print("max: {}, min {}".format(torch.max(grad).item(), torch.min(grad).item())))
        loss = -(mean_log_prob_pos / self.base_temperature) 
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    