import copy
import imp
import math
import os
import random
from symtable import Class
from turtle import pendown
from typing import Optional, Sequence
from urllib.request import proxy_bypass
from xml.sax.handler import feature_string_interning

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.dacs_transforms import get_class_masks, strong_transform
from helpers.matching_utils import (
    estimate_probability_of_confidence_interval_of_mixture_density, warp)
from helpers.metrics import MyMetricCollection
from helpers.utils import colorize_mask, resolve_ckpt_dir
from PIL import Image
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, instantiate_class

from .heads.base import BaseHead
from .modules import DropPath
from .prototype.prototype_generate import Prototype

@MODEL_REGISTRY
class DomainAdaptationSegmentationModel(pl.LightningModule):
    def __init__(self,
                 optimizer_init: dict,
                 lr_scheduler_init: dict,
                 backbone: nn.Module,
                 head: BaseHead,
                 dat: nn.Module,
                 loss: nn.Module,
                 prototype: Prototype,
                 pcl: nn.Module,
                 metrics: dict = {},
                 backbone_lr_factor: float = 1.0,
                 dar_lr_factor: float =1.0,
                 use_ref: bool = False,
                 adapt_to_ref: bool = False,
                 gamma: float = 0.25,
                 ema_momentum: float = 0.999,
                 pseudo_label_threshold: float = 0.968,
                 psweight_ignore_top: int = 15,
                 psweight_ignore_bottom: int = 120,
                 enable_fdist: bool = True,
                 fdist_lambda: float = 0.005,
                 fdist_classes: list = [6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
                 fdist_scale_min_ratio: float = 0.75,
                 color_jitter_s: float = 0.2,
                 color_jitter_p: float = 0.2,
                 blur: bool = True,
                 pretrained: Optional[str] = None,
                 ):
        super().__init__()

        #### MODEL ####
        # segmentation
        self.backbone = backbone
        self.head = head
        # DAT
        self.dat =dat
        # ema teacher network
        self.m_backbone = copy.deepcopy(self.backbone)
        self.m_head = copy.deepcopy(self.head)
        self.m_dat = copy.deepcopy(self.dat)
        for param in self.ema_parameters():
            param.requires_grad = False
        # imnet
        self.enable_fdist = enable_fdist
        if self.enable_fdist:
            self.imnet_backbone = copy.deepcopy(self.backbone)
            for param in self.imnet_backbone.parameters():
                param.requires_grad = False

        #### LOSSES ####
        # PixelWeightedCrossEntropyLoss
        self.loss = loss

        #### METRICS ####
        val_metrics = {'val_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('val', {}).items() for el in m}
        test_metrics = {'test_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('test', {}).items() for el in m}
        # {'val_DarkZurich_IoU': IoU()}
        self.valid_metrics = MyMetricCollection(val_metrics)
        # {'test_DarkZurich_IoU': IoU(), 'test_NighttimeDriving_IoU': IoU(), 'test_BDD100kNight_IoU': IoU()}
        self.test_metrics = MyMetricCollection(test_metrics)

        #### OPTIMIZATION ####
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.backbone_lr_factor = backbone_lr_factor
        self.dar_lr_factor = dar_lr_factor

        #### DAT ####
        self.adapt_to_ref = adapt_to_ref
        self.gamma = gamma
        self.use_ref = use_ref
        self.ref_out_estimator = prototype
        self.ref_out_estimator.Proto=self.ref_out_estimator.Proto.to(torch.device("cuda:1"))
        self.pcl = pcl

        #### OTHER STUFF ####
        self.ema_momentum = ema_momentum
        self.pseudo_label_threshold = pseudo_label_threshold
        self.psweight_ignore_top = psweight_ignore_top
        self.psweight_ignore_bottom = psweight_ignore_bottom
        self.fdist_lambda = fdist_lambda
        self.fdist_classes = fdist_classes
        self.fdist_scale_min_ratio = fdist_scale_min_ratio
        self.color_jitter_s = color_jitter_s
        self.color_jitter_p = color_jitter_p
        self.blur = blur
        # Manual Optimization
        self.automatic_optimization = False
        #### LOAD WEIGHTS ####
        self.load_weights(pretrained)

    def training_step(self, batch, batch_idx):
        """
        the complete training loop
        """
        # access your optimizers (one or multiple)
        opt = self.optimizers()
        # access any learning rate schedulers defined in your configure_optimizers().
        sch = self.lr_schedulers()
        # clear the gradients from the previous training step
        opt.zero_grad()
        # 更新teacher网络的参数
        self.update_momentum_encoder()
        #
        # SOURCE
        images_src, gt_src, image_enhance, semantic_enhance = batch['image_src'], batch['semantic_src'], batch['image_enhance'], batch['semantic_enhance']
        with torch.no_grad():
            semantic_enhance = semantic_enhance[:,0,:,:]
            temp = torch.sum(image_enhance,dim=1)
            semantic_enhance[temp==0] = 255
            semantic_enhance = semantic_enhance.to(torch.long)
        #print(semantic_enhance.shape)
        logits_src, feats_src = self.forward(images_src, return_feats=True) 
        loss_src = self.loss(logits_src, gt_src)
        self.log("train_loss_src", loss_src)
        # Keep the computational graph and allow back propagation again
        self.manual_backward(loss_src, retain_graph=self.enable_fdist)

        # ImageNet feature distance
        # DAFormer FD Loss
        if self.enable_fdist:
            loss_featdist_src = self.calc_feat_dist(
                images_src, gt_src, feats_src)
            self.log("train_loss_featdist_src", loss_featdist_src)
            self.manual_backward(loss_featdist_src)
        #
        #Source Argument
        #
        logits_enhance = self.forward(image_enhance)
        loss_enhance = self.loss(logits_enhance, semantic_enhance)
        self.log("train_loss_enhance", loss_enhance)
        # Keep the computational graph and allow back propagation again
        self.manual_backward(loss_enhance)
        with torch.no_grad():
            feats_src = self.backbone(images_src)
            feats_enhance = self.backbone(image_enhance)
        for i in range(4):
            feats_enhance[i].detach()
            feats_src[i].detach()
        logits_ref = self.ref_forward(feats_enhance,feats_src,image_enhance.shape)
        loss_ref = self.loss(logits_ref, semantic_enhance)
        B_ref, C_ref, H_ref, W_ref = logits_ref.shape
        _, ref_mask = torch.max(logits_ref, dim=1)
        semantic_enhance_reshape = F.interpolate(semantic_enhance.unsqueeze(0).float(), size=(H_ref, W_ref), mode='nearest').squeeze(0).long()
        semantic_enhance_reshape = semantic_enhance_reshape.contiguous().view(B_ref * H_ref * W_ref, )
        logits_ref = logits_ref.permute(0,2,3,1).contiguous().view(-1,C_ref)
        self.ref_out_estimator.update(features=logits_ref.detach(), labels=semantic_enhance_reshape.detach())
        
        ref_mask = ref_mask.contiguous().view(B_ref * H_ref * W_ref, )
        loss_source_prototype = self.pcl(Proto=self.ref_out_estimator.Proto.detach(), feat=logits_ref, labels=ref_mask)
        # freeze head's parameters
        for p in self.head.parameters():
            p.requires_grad = False
        self.log("train_loss_ref", loss_ref)
        self.log("train_source_prototype", loss_source_prototype)
        loss = loss_ref + loss_source_prototype
        self.manual_backward(loss)
        #
        #Adapatation
        #
        with torch.no_grad():
            feats_ref = self.backbone(batch['image_ref'])
            feats_tgt = self.backbone(batch['image_trg'])
        for i in range(4):
            feats_ref[i].detach()
            feats_tgt[i].detach()
        logits_tgt_ref = self.ref_forward(feats_tgt,feats_ref,batch['image_ref'].shape)
        _, tgt_ref_mask = torch.max(logits_tgt_ref, dim=1)
        tgt_ref_mask = tgt_ref_mask.contiguous().view(B_ref * H_ref * W_ref, )
        logits_tgt_ref = logits_tgt_ref.permute(0,2,3,1).contiguous().view(-1,C_ref)

        loss_tgt_prototype = self.pcl(Proto=self.ref_out_estimator.Proto.detach(), feat=logits_tgt_ref, labels=tgt_ref_mask)
        self.log("train_tgt_prototype", loss_tgt_prototype)
        self.manual_backward(loss_tgt_prototype)
        # unfreeze
        for p in self.head.parameters():
            p.requires_grad = True
        #
        # TARGET
        #
        with torch.no_grad():
            if self.adapt_to_ref and random.random() < 0.5:
                adapt_to_ref = True
                images_trg = batch['image_ref']
            else:
                adapt_to_ref = False
                images_trg = batch['image_trg']
            if self.use_ref and not adapt_to_ref:
                images_ref = batch['image_ref']
                feats_tgt = self.m_backbone(images_trg)
                feats_ref = self.m_backbone(images_ref)
                m_logits_trg = self.m_head(self.m_dat(feats_tgt, feats_ref)[0])
                m_logits_trg = nn.functional.interpolate(
                    m_logits_trg, size=images_trg.shape[-2:], mode='bilinear', align_corners=False)
                m_probs_trg = nn.functional.softmax(m_logits_trg, dim=1)
            else:
                """
                适应到reference domain,教师网络预测的作为伪标签
                """
                m_logits_trg = self.m_head(self.m_backbone(images_trg))
                m_logits_trg = nn.functional.interpolate(
                    m_logits_trg, size=images_trg.shape[-2:], mode='bilinear', align_corners=False)
                m_probs_trg = nn.functional.softmax(m_logits_trg, dim=1)
            # Get DACS-enhanced images
            # The student network is trained on the enhanced images
            # The teacher network generates pseudo labels using the unenhanced images
            mixed_img, mixed_lbl, mixed_weight = self.get_dacs_mix(
                images_trg, m_probs_trg, images_src, gt_src)

        # Train on mixed images
        mixed_pred = self.forward(mixed_img)
        mixed_loss = self.loss(mixed_pred, mixed_lbl,
                               pixel_weight=mixed_weight)
        self.log("train_loss_uda_trg", mixed_loss)
        # Backpropagating gradients
        self.manual_backward(mixed_loss)
        # update your model parameters
        opt.step()
        # be called at arbitrary intervals by the user in case of manual optimization, 
        # or by Lightning if "interval" is defined in configure_optimizers() in case of automatic optimization
        sch.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        the complete validation loop
        """
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['val'][dataloader_idx]
        for k, m in self.valid_metrics.items():
            if src_name in k:
                m(y_hat, y)

    def validation_epoch_end(self, outs):
        """
        Operations after a validation_epoch
        """
        out_dict = self.valid_metrics.compute()
        self.valid_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        the complete test loop
        """
        x, y = batch['image'], batch['semantic']
        y_hat = self.forward(x, out_size=y.shape[-2:])
        src_name = self.trainer.datamodule.idx_to_name['test'][dataloader_idx]
        for k, m in self.test_metrics.items():
            if src_name in k:
                m(y_hat, y)

    def test_epoch_end(self, outs):
        out_dict = self.test_metrics.compute()
        print(out_dict)
        import pdb
        pdb.set_trace()
        self.test_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        the complete prediction loop
        """
        dataset_name = self.trainer.datamodule.predict_on[dataloader_idx]
        save_dir = os.path.join(os.path.dirname(
            resolve_ckpt_dir(self.trainer)), 'preds', dataset_name)
        col_save_dir = os.path.join(os.path.dirname(
            resolve_ckpt_dir(self.trainer)), 'color_preds', dataset_name)
        if self.trainer.is_global_zero:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(col_save_dir, exist_ok=True)
        img_names = batch['filename']
        x = batch['image']
        orig_size = self.trainer.datamodule.predict_ds[dataloader_idx].orig_dims
        y_hat = self.forward(x, orig_size)
        preds = torch.argmax(y_hat, dim=1)
        for pred, im_name in zip(preds, img_names):
            arr = pred.cpu().numpy()
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(save_dir, im_name))
            col_image = colorize_mask(image)
            col_image.save(os.path.join(col_save_dir, im_name))

    def forward(self, x, out_size=None, return_feats=False):
        """
        Same as torch.nn.Module.forward().
        """
        feats = self.backbone(x)
        logits = self.head(feats)
        logits = nn.functional.interpolate(
            logits, x.shape[-2:], mode='bilinear', align_corners=False)
        if out_size is not None:
            logits = nn.functional.interpolate(
                logits, size=out_size, mode='bilinear', align_corners=False)
        if return_feats:
            return logits, feats
        return logits
    

    def ref_forward(self, feat_tgt, feat_ref,shape, out_size=None, return_feats=False):
        """
        Same as torch.nn.Module.forward().
        """
        feats, _, _ = self.dat(feat_tgt, feat_ref)
        logits = self.head(feats)
        logits = nn.functional.interpolate(
            logits, shape[-2:], mode='bilinear', align_corners=False)
        if return_feats:
            return logits, feats
        return logits

    def configure_optimizers(self):
        """"
        define optimizers and LR schedulers
        """
        # takes care of importing the class defined in class_path and instantiating it using some positional arguments,
        # in this case self.parameters(), and the init_args.
        optimizer = instantiate_class(
            self.optimizer_parameters(), self.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        return [optimizer], [lr_scheduler]

    def optimizer_parameters(self):
        backbone_weight_params = []
        backbone_bias_params = []
        head_weight_params = []
        head_bias_params = []
        dat_weight_params = []
        dat_bias_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('backbone'):
                if len(p.shape) == 1:  # bias and BN params
                    backbone_bias_params.append(p)
                else:
                    backbone_weight_params.append(p)
            elif name.startswith('head'):
                if len(p.shape) == 1:  # bias and BN params
                    head_bias_params.append(p)
                else:
                    head_weight_params.append(p)
            else:
                if len(p.shape) == 1:  # bias and BN params
                    dat_bias_params.append(p)
                else:
                    dat_weight_params.append(p)
        lr = self.optimizer_init['init_args']['lr']
        weight_decay = self.optimizer_init['init_args']['weight_decay']
        return [
            {'name': 'head_weight', 'params': head_weight_params,
                'lr': lr, 'weight_decay': weight_decay},
            {'name': 'head_bias', 'params': head_bias_params,
                'lr': lr, 'weight_decay': 0},
            {'name': 'backbone_weight', 'params': backbone_weight_params,
                'lr': self.backbone_lr_factor * lr, 'weight_decay': weight_decay},
            {'name': 'backbone_bias', 'params': backbone_bias_params,
                'lr': self.backbone_lr_factor * lr, 'weight_decay': 0},
            {'name': 'dat_weight', 'params': dat_weight_params,
                'lr': self.dar_lr_factor * lr, 'weight_decay': weight_decay},
            {'name': 'dat_bias', 'params': dat_bias_params,
                'lr': self.dar_lr_factor * lr, 'weight_decay': 0}
        ]

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=self.device)
        elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrain_path)):
            checkpoint = torch.load(os.path.join(os.environ.get(
                'TORCH_HOME', ''), 'hub', pretrain_path), map_location=self.device)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrain_path, progress=True, map_location=self.device)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=True)


    @staticmethod
    @torch.no_grad()
    def eta(logits):  # normalized entropy / efficiency
        dim = logits.shape[1]
        p_log_p = nn.functional.softmax(
            logits, dim=1) * nn.functional.log_softmax(logits, dim=1)
        ent = -1.0 * p_log_p.sum(dim=1)  # b x h x w
        return ent / math.log(dim)


    @torch.no_grad()
    def get_dacs_mix(self, images_trg, probs_trg, images_src, gt_src):
        """
        进行DACS的mix,相当于一种另类的增强
        """
        # take first source images in batch
        src_images_bs = images_src.shape[0]
        trg_images_bs = images_trg.shape[0]
        if src_images_bs > trg_images_bs:
            images_src = images_src[:trg_images_bs]
            gt_src = gt_src[:trg_images_bs]

        images_bs = images_trg.shape[0]
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
        }

        trg_pseudo_prob, trg_pseudo_label = torch.max(probs_trg, dim=1)
        trg_ps_large_p = trg_pseudo_prob.ge(
            self.pseudo_label_threshold).long() == 1
        trg_ps_size = torch.numel(trg_pseudo_label)
        trg_pseudo_weight = torch.sum(trg_ps_large_p) / trg_ps_size
        trg_pseudo_weight = torch.full_like(trg_pseudo_prob, trg_pseudo_weight)
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            trg_pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            trg_pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(
            (trg_pseudo_weight.shape), device=self.device)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * images_bs, [None] * images_bs
        mix_masks = get_class_masks(gt_src.unsqueeze(1))

        for i in range(images_bs):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((images_src[i], images_trg[i])),
                target=torch.stack((gt_src[i], trg_pseudo_label[i])))
            _, trg_pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], trg_pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl).squeeze(1)
        return mixed_img, mixed_lbl, trg_pseudo_weight


    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            feat_imnet = self.imnet_backbone(img)
            if isinstance(feat_imnet, Sequence):
                feat_imnet = [f.detach() for f in feat_imnet]
            else:
                feat_imnet = [feat_imnet.detach()]
                if feat is not None:
                    feat = [feat]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = self.downscale_label_ratio(gt.unsqueeze(1), scale_factor,
                                                     self.fdist_scale_min_ratio,
                                                     self.head.num_classes,
                                                     255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_loss = self.fdist_lambda * feat_dist
        return feat_loss

    @staticmethod
    def masked_feat_dist(f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
        return torch.mean(pw_feat_dist)

    @staticmethod
    def downscale_label_ratio(gt,
                              scale_factor,
                              min_ratio,
                              n_classes,
                              ignore_index=255):
        assert scale_factor > 1
        bs, orig_c, orig_h, orig_w = gt.shape
        assert orig_c == 1
        trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
        ignore_substitute = n_classes

        out = gt.clone()  # otw. next line would modify original gt
        out[out == ignore_index] = ignore_substitute
        out = nn.functional.one_hot(
            out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, n_classes +
                                   1, orig_h, orig_w], out.shape
        out = nn.functional.avg_pool2d(out.float(), kernel_size=scale_factor)
        gt_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == ignore_substitute] = ignore_index
        out[gt_ratio < min_ratio] = ignore_index
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out

    def ema_parameters(self):
        """
        获取teacher网络的参数
        """
        for m in filter(None, [self.m_backbone, self.m_head, self.m_dat]):
            for p in m.parameters():
                yield p

    def live_parameters(self):
        """
        获取student网络的参数
        """
        for m in filter(None, [self.backbone, self.head, self.dat]):
            for p in m.parameters():
                yield p

    @torch.no_grad()
    def update_momentum_encoder(self):
        """
        指数加权平均
        """
        m = min(1.0 - 1 / (float(self.global_step) + 1.0),
                self.ema_momentum)  # limit momentum in the beginning
        for param, param_m in zip(self.live_parameters(), self.ema_parameters()):
            if not param.data.shape:
                param_m.data = param_m.data * m + param.data * (1. - m)
            else:
                param_m.data[:] = param_m[:].data[:] * \
                    m + param[:].data[:] * (1. - m)
    
    def train(self, mode=True):
        super().train(mode=mode)
        for m in filter(None, [self.m_backbone, self.m_head]):
            if isinstance(m, nn.modules.dropout._DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if self.enable_fdist:  # always in eval mode
            self.imnet_backbone.eval()
    

    @torch.no_grad()
    def refine(self, logits_trg, logits_aligned, warp_mask=None, certs=None):
        c = logits_trg.shape[1]
        assert c == 19, 'we assume cityscapes classes'

        probs_trg = nn.functional.softmax(logits_trg, dim=1)
        probs_aligned = nn.functional.softmax(logits_aligned, dim=1)
        # trust score s
        s = torch.mean(self.eta(logits_trg), dim=(1, 2)) ** self.gamma  # b
        self.log("trust_score",s.mean())
        epsilon = s[:,None,None,None]
        probs_trg_refined = (1 - epsilon) * probs_trg + epsilon * probs_aligned

        return probs_trg_refined
