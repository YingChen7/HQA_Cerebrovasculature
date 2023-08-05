# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:16:19 2021

@author: CY
"""
from typing import Callable
import time
from tqdm import tqdm
import copy
import torch
from nnunet.training.ssl.base import Model
from nnunet.training.data_augmentation.data_augmentation_batch import get_batch_augmentation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SelfSupervisedModel(Model):
    def __init__(self, network: torch.nn.Module,
                 dataloader,
                 optimizer: torch.optim,
                 criterion: Callable = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 data_aug_params = None,
                 patch_size = None,
                 additional_identifier: str = ''):
        super().__init__(network, dataloader, optimizer, criterion, lr_scheduler, additional_identifier)
        self.metrices = []
        self.data_aug_params = data_aug_params
        self.patch_size = patch_size
        if criterion is None:
            self.criterion = torch.nn.MSELoss
            
    def train(self, num_epochs: int):
        # 覆盖父类的train函数
        since = time.time()
        losses = {'train': [], 'val': []}
        stats = {'train': [], 'val': []}
        lr_rates = []
        best_epoch, best_stat = None, float('inf')
        best_model_weights = None
        self.network.to(device)
        self.network.initialize_target_network()      
        train_max_step = 500
        val_max_step = 50
        for epoch in tqdm(range(num_epochs)):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.network.train()
                else:
                    self.network.eval()
                running_loss = 0.0
                
                # for i, (imgs_dict, tf_imgs_dict) in enumerate(zip(self.dataloader[phase][0], self.dataloader[phase][1])):
                cnt = 0
                for imgs_dict in self.dataloader[phase]:
                    cnt  = cnt + 1
                    imgs = imgs_dict.get('data')
                    imgs, tf_imgs = get_batch_augmentation(imgs, self.patch_size, self.data_aug_params)
                    imgs, tf_imgs = imgs.to(device), tf_imgs.to(device)
                    self.optim.zero_grad()   
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        online_feats = self.network.online_network(imgs)
                        tf_online_feats = self.network.online_network(tf_imgs)

                        pred1_feats = self.network.predictor(self.network.online_projector(online_feats))
                        pred2_feats = self.network.predictor(self.network.online_projector(tf_online_feats))
                    
                    with torch.no_grad():
                        target_feats = self.network.target_network(imgs)
                        target_feats_tf = self.network.target_network(tf_imgs)

                        target_for_pred1_feats = self.network.target_projector(target_feats_tf)
                        target_for_pred2_feats = self.network.target_projector(target_feats)
                        
                    with torch.set_grad_enabled(phase == 'train'):
                        loss = self.network.regression_loss(pred1_feats, target_for_pred1_feats)
                        loss += self.network.regression_loss(pred2_feats, target_for_pred2_feats)
                        # print(phase + ':', loss)
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()
                            self.network.update_target_network()
                    running_loss += loss.item() * imgs.size(0)
                    
                    if phase == 'train' and cnt >= train_max_step:
                        epoch_loss = running_loss / (train_max_step * 2)
                        break
                    if phase == 'val' and cnt >= val_max_step:
                        epoch_loss = running_loss / (val_max_step * 2)
                        break
                
                if phase == 'val' and epoch_loss <= best_stat:
                    best_epoch = epoch
                    best_stat = epoch_loss
                    best_model_weights = copy.deepcopy(self.network.state_dict())
                losses[phase].append(epoch_loss)

            self.log_stats(epoch, losses) 
            
            if (epoch + 1) % 30 == 0: #checkpoint model every 30 epochs
                torch.save(self.network.state_dict(), self.weights_file_name)
                torch.save(best_model_weights, self.weights_file_name.split('.')[0]+'_best.h5')
                print('model checkpointed to ', self.weights_file_name)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_loss)  # considering ReduceLrOnPlateau and watch val val loss
                lr_step = self.optim.state_dict()["param_groups"][0]["lr"]
                lr_rates.append(lr_step)
                min_lr = self.lr_scheduler.min_lrs[0]
                if lr_step / min_lr <= 10:
                    print("Min LR reached")
                    break
                
        time_elapsed = time.time() - since
        print('Training Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss {:.6f} and best epoch is {}'.format(best_stat, best_epoch + 1))
        torch.save(self.network.state_dict(), self.weights_file_name)
        print('model saved to ', self.weights_file_name)
        self.network.load_state_dict(best_model_weights)
                        
                        