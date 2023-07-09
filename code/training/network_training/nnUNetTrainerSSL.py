#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
from collections import OrderedDict
from typing import Tuple
from torch.optim.optimizer import Optimizer
import numpy as np
import time
import torch.distributed as dist
from torch.backends import cudnn
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from shutil import copyfile

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2, MultOut_SameTarget_Loss
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet_pre import Generic_UNet_pre
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.dataloading.SSLDataLoader import SSLDataLoader
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.ssl.VessPro_multi import VessPro
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Splits param group into weight_decay / non-weight decay.
       Tweaked from https://bit.ly/3dzyqod
    :param model: the torch.nn model
    :param weight_decay: weight decay term
    :param skip_list: extra modules (besides BN/bias) to skip
    :returns: split param group into weight_decay/not-weight decay
    :rtype: list(dict)
    """
    # if weight_decay == 0:
    #     return model.parameters()

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay':  0, 'ignore': True},
        {'params': decay, 'weight_decay': weight_decay, 'ignore': False}]

class LARS(Optimizer):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.
    __ : https://arxiv.org/abs/1708.03888
    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::
    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate
    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    """

    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        if eps < 0.0:
            raise ValueError('invalid epsilon value: , %f' % eps)

        if trust_coef < 0.0:
            raise ValueError("invalid trust coefficient: %f" % trust_coef)

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef

    def __getstate__(self):
        lars_dict = {}
        lars_dict['eps'] = self.eps
        lars_dict['trust_coef'] = self.trust_coef
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state
        self.eps = lars_dict['eps']
        self.trust_coef = lars_dict['trust_coef']

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def state(self):
        return self.optim.state

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def apply_adaptive_lrs(self):
        with torch.no_grad():
            for group in self.optim.param_groups:
                weight_decay = group['weight_decay']
                ignore = group.get('ignore', None)  # NOTE: this is set by add_weight_decay

                for p in group['params']:
                    if p.grad is None:
                        continue

                    # Add weight decay before computing adaptive LR
                    # Seems to be pretty important in SIMclr style models.
                    if weight_decay > 0:
                        p.grad = p.grad.add(p, alpha=weight_decay)

                    # Ignore bias / bn terms for LARS update
                    if ignore is not None and not ignore:
                        # compute the parameter and gradient norms
                        param_norm = p.norm()
                        grad_norm = p.grad.norm()

                        # compute our adaptive learning rate
                        adaptive_lr = 1.0
                        if param_norm > 0 and grad_norm > 0:
                            adaptive_lr = self.trust_coef * param_norm / (grad_norm + self.eps)

                        # print("applying {} lr scaling to param of shape {}".format(adaptive_lr, p.shape))
                        p.grad = p.grad.mul(adaptive_lr)

    def step(self, *args, **kwargs):
        self.apply_adaptive_lrs()

        # Zero out weight decay as we do it in LARS
        weight_decay_orig = [group['weight_decay'] for group in self.optim.param_groups]
        for group in self.optim.param_groups:
            group['weight_decay'] = 0

        loss = self.optim.step(*args, **kwargs)  # Normal optimizer

        # Restore weight decay
        for group, wo in zip(self.optim.param_groups, weight_decay_orig):
            group['weight_decay'] = wo

        return loss

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class nnUNetTrainerSSL(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        # initialize 
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            if self.params['pretrained_model']:
                 assert os.path.isfile(self.params['pretrained_model'])
                 self.load_pretrained()

            if self.params['auto_resume']:
                resume_file = os.path.join(self.output_folder, "current.pth")
                if os.path.exists(resume_file):
                    self.print_to_log_file(f'auto resume from {resume_file}')
                    self.params['resume'] = resume_file
                else:
                    self.print_to_log_file(f'no checkpoint found in {self.output_folder}, ignoring auto resume')
            if self.params['resume']:
                assert os.path.isfile(self.params['resume'])
                self.load_checkpoint()
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
        
        self.summary_writer = SummaryWriter(log_dir=self.output_folder)
    def get_basic_generators(self):
        self.load_dataset()
        
        dl_tr =SSLDataLoader(self.dataset, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 memmap_mode='r')
        
        return dl_tr

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.backbone= Generic_UNet_pre(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.backbone.inference_apply_nonlin = softmax_helper
        self.params = {'VessPro_p': 2, 'VessPro-momentum': 0.7, 'VessPro-pos-ratio': 0.7, 'VessPro-ins-loss-weight': 0.1, 
                       'VessPro-clamp-value': 0.,  'VessPro_transform_layer': 1, 
                       'encoder_final_channels': self.backbone.final_num_features, 
                       'encoder_channels': [32, 64, 128, 256, 320],
                       'step_size': 400, 'batch_size': self.batch_size,
                       'epochs': 400, 'start_epoch': 1, 'weight_decay': 1e-5,
                       'base_learning_rate':0.5, 'momentum': 0.9,
                       'amp_opt_level': 'O1', 'warmup_epoch': 5,
                       'warmup_multiplier': 100, 'pretrained_model': None,
                       'auto_resume': False, 'print_freq':100,
                       'save_freq': 2, 'resume': None
            }
        
        path = os.path.join(self.output_folder, "config.json")
        with open(path, 'w') as f:
            json.dump(self.params, f, indent=2)
        self.print_to_log_file("Full config saved to {}".format(path))
        self.print_to_log_file(
             "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(self.params.items()))
         )
         
        self.network = VessPro(self.backbone, self.params)
        if torch.cuda.is_available():
            self.network.cuda()
        
        self.training_params = add_weight_decay(self.network, self.params['weight_decay'])
        self.print_to_log_file(self.network)
        self.print_to_log_file("\n")
        
        

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

        self.optimizer = torch.optim.SGD(
            self.training_params,
            lr=self.params['base_learning_rate'],
            momentum=self.params['momentum'])
        self.optimizer = LARS(self.optimizer)

        self.lr_scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            eta_min=0.000001,
            T_max=(self.params['epochs'] - self.params['warmup_epoch']) * self.params['step_size'])
        
        if self.params['warmup_epoch']>0:
            self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier= self.params['warmup_multiplier'],
            after_scheduler=self.lr_scheduler,
            warmup_epoch= self.params['warmup_epoch'] * self.params['step_size'])
        
    def load_pretrained(self):
        ckpt = torch.load(self.params['pretrained_model'], map_location='cpu')
        state_dict = ckpt['model']
        model_dict = self.network.state_dict()
    
        model_dict.update(state_dict)
        self.network.load_state_dict(model_dict)
        self.print_to_log_file(f"==> loaded checkpoint '{self.params['pretrained_model']}' (epoch {ckpt['epoch']})")
        
    def load_checkpoint(self):
        self.print_to_log_file(f"=> loading checkpoint '{self.params['resume']}'")
    
        checkpoint = torch.load(self.params['resume'], map_location='cpu')
        self.params['start_epoch'] = checkpoint['epoch'] + 1
        self.nerwork.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
    
        if self.params['amp_opt_level'] != "O0" and checkpoint['opt'].amp_opt_level != "O0":
            amp.load_state_dict(checkpoint['amp'])
    
        self.print_to_log_file(f"=> loaded successfully '{self.params['resume']}' (epoch {checkpoint['epoch']})")
    
        del checkpoint
        torch.cuda.empty_cache()
        
    def save_checkpoint(self, epoch):
        self.print_to_log_file('==> Saving...')
        state = {
            'opt': self.params,
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        file_name = os.path.join(self.output_folder, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, file_name)
        copyfile(file_name, os.path.join(self.output_folder, 'current.pth'))
             
        
    def run_experiment(self):
        """
        config='{"dataset": "cityscapes", "model": "SelfSupervisedModel", "network": "BYOL", "mode": "self-supervised",
        "network_args":{"backbone":"Resnet50", "pretrained": false, "target_momentum": 0.996},
        "train_args":{"batch_size": 32, "epochs": 700, "log_to_tensorboard": false},
        "experiment_group":{}
        }'
    
        """
        
        for epoch in range(self.params['start_epoch'], self.params['epochs'] +1):
            
            if isinstance(self.dl_tr, DistributedSampler):
                self.dr_tr.sampler.set_epoch(epoch)
                
            self.run_epoch(epoch)
            
            if (epoch % self.params['save_freq'] == 0 or epoch == self.params['epochs']):
                self.save_checkpoint(epoch)
            
            
    def run_epoch(self, epoch):
        self.network.train()
            
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        end = time.time()
        for idx, (data1_dict, data2_dict) in enumerate((self.dl_tr)):
            img1 = data1_dict.get('data').cuda()
            coord1 = data1_dict.get('coord').cuda()
            img2 = data2_dict.get('data').cuda()
            coord2 = data2_dict.get('coord').cuda()
            
            loss = self.network(img1, img2, coord1, coord2)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            loss_meter.update(loss.item(), img1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % self.params['print_freq'] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.print_to_log_file(
                    f'Train: [{epoch}/{self.params["epochs"]}][{idx}/{self.params["step_size"]}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'lr {lr:.3f}  '
                    f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
    
                # tensorboard logger
                if self.summary_writer is not None:
                    step = (epoch - 1) * self.params['step_size'] + idx
                    self.summary_writer.add_scalar('lr', lr, step)
                    self.summary_writer.add_scalar('loss', loss_meter.val, step)
            if idx>=self.params['step_size']:
                break
            
            


    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)
    


    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        # do_ds: do deep supervision
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        subsource = data_dict['subsource']
        
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        subsource = maybe_to_torch(subsource)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            subsource = to_cuda(subsource)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output, classify = self.network(data)
                del data
                l = self.loss(output, target)
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output, classify = self.network(data)
            del data
            l = self.loss(output, target)
            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        elif self.fold == 0:
            all_keys = np.sort(list(self.dataset.keys()))
            all_HH_keys = [name for name in all_keys if 'HH' in name]
            all_Guys_keys = [name for name in all_keys if 'Guys' in name]
            all_IOP_keys = [name for name in all_keys if 'IOP' in name]
            all_AB_keys = [name for name in all_keys if 'AB' in name]
            print(len(all_HH_keys))
            print(len(all_Guys_keys))
            print(len(all_IOP_keys))
            print(len(all_AB_keys))
            tr_keys = all_HH_keys[:]+ all_Guys_keys[:] + all_IOP_keys[:] + all_AB_keys[:]
            val_keys = all_HH_keys[36:]+ all_Guys_keys[52:] + all_IOP_keys[48:] + all_AB_keys[111:]
            print(tr_keys)
            print(val_keys)
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
