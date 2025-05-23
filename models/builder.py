import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
# from utils.load_utils import load_pretrain
# from functools import partial

from utils.engine.logger import get_logger

logger = get_logger()


class FewShotSegmentation(nn.Module):
    def __init__(self, seg_model):
        super(FewShotSegmentation, self).__init__()
        self.model = seg_model
    
    def forward(self, s_imgs, s_depths, s_masks, q_imgs, q_depths):
        if s_masks.size()[1] != 1:
            _, _, H, W = s_masks.size()
            s_masks = s_masks.view(-1, 1, H, W)
        # Extract support features
        s_features = self.model.encode(s_imgs, s_depths)  # [B, C, H, W]
        q_features = self.model.encode(q_imgs, q_depths)      # [B, C, H, W]
        
        # Compute class prototypes
        _, _, H2, W2 = s_features[-1].size()
        s_masks = F.interpolate(s_masks, size=(H2, W2), mode='bilinear', align_corners=False)
        prototypes = self.compute_prototypes(s_features[-1], s_masks)  # [classes, B*shots, classes]
        
        # Compute similarity and classify query pixels
        # matched q_features at layer 4
        q_out4 = self.match_prototypes(q_features[-1], prototypes)  # [B, 15, 20]
        # def decode(self, x, rgb):
        q_logits = self.model.decode(q_features, q_imgs)

        return q_out4, q_logits, prototypes
    
    def compute_prototypes(self, features, masks):
        """ Compute class-wise prototype vectors from support set. """
        # C = features.shape[1]  # Number of feature channels
        prototypes = []
        for c in range(2):  # Ignore background (255)
            mask = (masks == c).float()  # Binary mask for class c
            prototype = (features * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
            prototypes.append(prototype)
        return torch.stack(prototypes)  # [N_classes, C]
    
    def match_prototypes(self, q_features, prototypes):
        """ Compute similarity between query features and prototypes """
        B, C, H, W = q_features.shape
        q_features = q_features.view(B, C, -1)  # Flatten to [B, C, HW]
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(q_features.unsqueeze(1), prototypes.unsqueeze(-1), dim=2)
        q_logits = similarities.argmax(dim=1).view(B, H, W)  # Assign class with highest similarity
        return q_logits


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q_features, prototypes, q_masks):
        """ Compute contrastive loss: Query pixels should match their correct prototype. """
        if len(q_features.size()) == 3:
            B, H, W = q_features.size()
            q_features = q_features.view(B, 1, -1)
        else:
            B, C, H, W = q_features.shape
            q_features = q_features.view(B, C, -1)  # [B, C, HW]
        q_masks = q_masks.view(B, -1)  # [B, HW]
        
        similarities = F.cosine_similarity(q_features.unsqueeze(1), prototypes.unsqueeze(-1), dim=2)
        
        # Compute contrastive loss
        pos_mask = torch.zeros_like(similarities)  # [classes, B*shots, 15*20]
        for b in range(B):
            pos_mask[b, q_masks[b], :] = 1  # Mark positive pairs
        pos_similarity = (pos_mask * similarities).sum(dim=1) / pos_mask.sum(dim=1)
        neg_similarity = ((1 - pos_mask) * similarities).sum(dim=1) / (1 - pos_mask).sum(dim=1)
        
        loss = -torch.log(torch.exp(pos_similarity / self.temperature) /
                          (torch.exp(pos_similarity / self.temperature) +
                           torch.exp(neg_similarity / self.temperature)))
        return loss.mean()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='none', ignore_index=255), norm_layer=nn.BatchNorm2d, syncbn=False):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg
        
        if cfg.backbone == 'DFormer-Large':
            from .encoders.DFormer import DFormer_Large as backbone
            self.channels=[96, 192, 288, 576]
        elif cfg.backbone == 'DFormer-Base':
            from .encoders.DFormer import DFormer_Base as backbone
            self.channels=[64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Small':
            from .encoders.DFormer import DFormer_Small as backbone
            self.channels=[64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Tiny':
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels=[32, 64, 128, 256]
        elif cfg.backbone == 'DFormerTrav-Base':
            from .encoders.DFormer import get_DFormerTrav as backbone
            self.channels=[64, 128, 256, 512]

        if syncbn:
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        else:
            norm_cfg=dict(type='BN', requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)
        

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'ham': # True
            logger.info('Using Ham Decoder')
            print(cfg.num_classes)
            from .decoders.ham_head import LightHamHead as DecoderHead
            # from mmseg.models.decode_heads.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels[1:], num_classes=cfg.num_classes, in_index=[1,2,3],norm_cfg=norm_cfg, channels=cfg.decoder_embed_dim)
            from .decoders.fcnhead import FCNHead
            if cfg.aux_rate!=0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print('aux rate is set to',str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
            
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        elif cfg.decoder == 'nl':
            logger.info('Using Decoder: nl+')
            from .decoders.nl_head import NLHead as Head
            self.decode_head = Head(in_channels=self.channels[1:], in_index=[1,2,3],num_classes=cfg.num_classes, norm_cfg=norm_cfg,channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        # print('builder',rgb.shape,modal_x.shape)
        x = self.backbone(rgb, modal_x)
        if self.cfg.decoder == 'nl_near_far':
            out = self.decode_head.forward(x, modal_x=modal_x)
        else:  # True
            out = self.decode_head.forward(x)  # [B, 2, 60, 80]
        out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        if self.aux_head:  # None
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out  # [B, 2, 480, 640]
    
    def encode(self, rgb, modal_x):
        return self.backbone(rgb, modal_x)
    
    def decode(self, x, rgb):
        orisize = rgb.shape
        
        out = self.decode_head.forward(x)  # [B, 2, 60, 80]
        out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        if self.aux_head:  # None
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        # print('builder',rgb.shape,modal_x.shape)
        if self.aux_head:  # False
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:  # train
            loss = self.criterion(out, label.long())[label.long() != self.cfg.background].mean()
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())[label.long() != self.cfg.background].mean()
            return loss, out
        else:  # eval
            return out