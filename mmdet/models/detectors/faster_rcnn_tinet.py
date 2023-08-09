import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# from torch.utils.tensorboard import SummaryWriter

import numpy as np

import warnings

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector

EPSILON = 1e-5

class IGFW(nn.Module):
    def __init__(self, downsample_size=(64, 64)) -> None:
        super().__init__()
        self.downsample_size = downsample_size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        self.fc_block = nn.Sequential(
            nn.Linear(8192, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 2)
        )
        self.softmax = nn.Softmax(1)
        self.register_parameter(name='alpha', param=torch.nn.Parameter(torch.tensor(1.)))

    
    def forward(self, x):
        x = T.Resize(self.downsample_size)(x)
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc_block(x)
        x = self.softmax(x)
        p_d, p_n = x[:, 0], x[:, 1]
        w_v = 0.5*(p_d - p_n)*self.alpha + 0.5
        w_t = 1 - w_v

        return w_v, w_t, x, self.alpha


class InterMA(nn.Module):
    def __init__(self, in_channels, reduction=16) -> None:
        super().__init__()
        self.gap_v = nn.AdaptiveAvgPool2d(1)
        self.gap_t = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        self.res_v = ResidualBlock(in_channels, in_channels)
        self.res_t = ResidualBlock(in_channels, in_channels)


    def forward(self, v_feat, t_feat):

        d_feat = v_feat - t_feat
        d_feat2 = t_feat - v_feat
        # import pdb; pdb.set_trace()
        d_vector = self.fc(self.gap_v(d_feat).squeeze())
        d_vector2 = self.fc(self.gap_t(d_feat2).squeeze())
        
        vd_feat = v_feat*d_vector.unsqueeze(-1).unsqueeze(-1)
        td_feat = t_feat*d_vector2.unsqueeze(-1).unsqueeze(-1)

        v_feat_ = v_feat + td_feat
        t_feat_ = t_feat + vd_feat

        v_feat = v_feat + self.res_v(v_feat_)
        t_feat = t_feat + self.res_t(t_feat_)

        return v_feat, t_feat


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class IntraMA(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.predictor_v = MaskPredictor(in_channels)
        self.predictor_t = MaskPredictor(in_channels)
    
    def forward(self, v_feat, t_feat):
        m_vs, m_ts = [], []
        v_feat_new, t_feat_new = [], []
        for i in range(len(v_feat)):
            m_v = self.predictor_v(v_feat[i])
            m_t = self.predictor_t(t_feat[i])
            v_feat_ = v_feat[i]*(1 + m_v)
            t_feat_ = t_feat[i]*(1 + m_t)

            m_vs.append(m_v)
            m_ts.append(m_t)
            v_feat_new.append(v_feat_)
            t_feat_new.append(t_feat_)
        return v_feat_new, t_feat_new, m_vs, m_ts


class MaskPredictor(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        # self.conv = nn.Conv2d(in_channels, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)

        # x = self.relu(x)
        return x

def create_elipsis_mask(bboxes, img_shape, mode=1):
    bs, c, h, w = img_shape
    mask = torch.zeros((bs, h, w), device='cuda')
    for i, batch_bboxes in enumerate(bboxes):
        n = len(batch_bboxes)
        mask_ = torch.zeros((n, h, w))
        xmin, ymin, xmax, ymax = batch_bboxes[:, 0], batch_bboxes[:, 1], batch_bboxes[:, 2], batch_bboxes[:, 3]
        a = (xmax - xmin)/2
        b = (ymax - ymin)/2
        cx = (xmax + xmin)/2
        cy = (ymax + ymin)/2

        x = torch.tensor(np.arange(0, w, 1), device='cuda:0')
        y = torch.tensor(np.arange(0, h, 1), device='cuda:0')
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        for j in range(n):
            # binary mask
            if mode == 1:
                m_ = (xx-cx[j])**2/a[j]**2 + (yy-cy[j])**2/b[j]**2 < 1+EPSILON
                m_.to(dtype=torch.float32)
                mask_[j, :, :] = m_
            else:
                # gauss heatmap
                gauss_map = gaus2d(xx, yy, cx[j], cy[j], a[j], b[j])
                mask_[j, :, :] = gauss_map.clone().detach()


        mask[i, :, :] = mask_.sum(0).clamp_(min=0, max=1)
        # mask[i, :, :] = mask_.sum(0)

    return mask

def gaus2d(xx, yy, mux=0, muy=0, sigx=1, sigy=1, A=1):
    return A*torch.exp(-((xx - mux)**2. / (2. * sigx**2.) + (yy - muy)**2. / (2. * sigy**2.)))
    # return 1. / (2. * np.pi * sigx * sigy) * torch.exp(-((xx - mux)**2. / (2. * sigx**2.) + (yy - muy)**2. / (2. * sigy**2.)))


def illumination_loss(p, y, w):
    i_loss = nn.CrossEntropyLoss()(p, y)
    return dict(i_loss=i_loss*w)

def multi_level_mask_loss(ms, y, w=None, mode=1):
    m_loss = 0
    y = torch.unsqueeze(y, 1)
    for i, m in enumerate(ms):
        y_ = F.interpolate(y, size=m.shape[-2:])
        # mask: dice
        if mode == 1:
            m_loss = m_loss + dice_loss(m, y_, w)
        # heatmap: squared error
        else:
            m_loss = m_loss + F.mse_loss(m, y_)
    return m_loss/(i+1)

def dice_loss(predict, target, weight=None):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num = predict.size(0)

    pre = torch.sigmoid(predict).view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1)  # 利用预测值与标签相乘当作交集
    union = (pre + tar).sum(-1)

    score = 1 - 2 * (intersection + EPSILON) / (union + EPSILON)
    if weight is not None:
        score = score*weight

    score.mean()
    return score


def save_feature_to_img(features, name, timestamp, channel=None, output_dir=None, method='matshow'):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = '/home/zhangyan22/mmdetection/work_dir/faster_rcnn_r50_fpn_tinet/flir2/vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)

    # for i in range(len(features)):
    if isinstance(features, list) or isinstance(features, tuple):
        for i in range(3):
            features_ = features[i]
            for j in range(features_.shape[0]):
                upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                features_ = upsample(features_)

                feature = features_[j, :, :, :]
                if channel is None:
                    feature = torch.sum(feature, 0)
                else:
                    feature = feature[channel, :, :]
                feature = feature.detach().cpu().numpy() # 转为numpy

                dist_dir = os.path.join(output_dir, timestamp)
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                if method == 'cv2':
                    img = 255-(feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                    img = img.astype(np.uint8)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(dist_dir, name + str(i) + '.jpg'), img)
                else:
                    plt.matshow(feature, interpolation='nearest')
                    plt.colorbar()
                    plt.axis('off')
                    plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                    plt.close()
    else:
        for j in range(features.shape[0]):
            upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            features = upsample(features)

            feature = features[j, :, :, :]
            if channel is None:
                feature = torch.sum(feature, 0)
            else:
                feature = feature[channel, :, :]
            feature = feature.detach().cpu().numpy() # 转为numpy

            if method == 'cv2':
                img = 255-(feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(dist_dir, name + str(i) + '.jpg'), img)
            else:
                plt.matshow(feature, interpolation='nearest')
                plt.colorbar()
                plt.axis('off')
                plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                plt.close()


def plot_distri(x, name, timestamp, output_dir=None):
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = '/home/zhangyan22/mmdetection/work_dir/faster_rcnn_r50_fpn_tinet/flir/vis_test'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dist_dir = os.path.join(output_dir, timestamp)
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)
    
    x = torch.flatten(x)
    x = x.detach().cpu().numpy()
    plt.plot(np.sort(x))
    plt.savefig(os.path.join(dist_dir, name+'.png'))
    plt.close()



@DETECTORS.register_module()
class TINet(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 igfw=None,
                 inter=None,
                 intra=None,
                 weighted_loss=None,
                 w=None,
                 pretrained=None,
                 init_cfg=None):
        super(TINet, self).__init__(backbone)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        # self.backbone = build_backbone(backbone)
        self.backbone_v = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone)

        self.weighted_loss = weighted_loss

        if w is not None:
            self.w_i = w['i']
            self.w_m = w['m']
        else:
            self.w_i = 1
            self.w_m = 1

        in_channels = neck['out_channels']
        
        if inter is not None:
            if inter == 1:
                self.inter_ma1 = InterMA(in_channels)
                self.inter_ma2 = InterMA(in_channels)
                self.inter_ma3 = InterMA(in_channels)
                self.inter_ma4 = InterMA(in_channels)
                self.inter_ma5 = InterMA(in_channels)

            self.inter_ma = [self.inter_ma1, self.inter_ma2, self.inter_ma3, self.inter_ma4, self.inter_ma5]
            # self.inter_ma = [InterMA(in_channels).to('cuda:0') for i in range(neck['num_outs'])]
        else:
            self.inter_ma = None

        if intra is not None:
            self.intra_ma = IntraMA(in_channels)
            self.intra_mode = intra
        else:
            self.intra_ma = None

        if igfw is not None:
            self.igfw = IGFW(igfw['downsample_size'])
        else:
            self.igfw = None
        
        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.writer = SummaryWriter()
        self.iteration = 0
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        v_img, t_img = img
        v_feat = self.backbone_v(v_img)
        t_feat = self.backbone_t(t_img)

        # v_feat, t_feat, v_feat_s, t_feat_s = self.backbone(img)
        if self.with_neck:
            v_feat = self.neck(v_feat)
            t_feat = self.neck(t_feat)

        return v_feat, t_feat


    def set_epoch(self, epoch):
        self.epoch = epoch + 1

    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        v_feat, t_feat = self.extract_feat(img)

        losses = dict()

        n_level = len(v_feat)

        self.iteration += 1

        # -------------illumination-guided weighting module---------------
        if self.igfw is not None:
            w_v, w_t, p_dn, alpha = self.igfw(img[0])
            
            y_dn = [img_meta['dn_labels'] for img_meta in img_metas]
            y_dn = p_dn.new_tensor(y_dn)
            i_loss = illumination_loss(p_dn, y_dn, self.w_i)
            losses.update(i_loss)

        else:
            w_v = torch.tensor(0.5, dtype=torch.float32, device='cuda:0')
            w_t = torch.tensor(0.5, dtype=torch.float32, device='cuda:0')

        # -------------------------Inter-MA------------------------------
        if self.inter_ma is not None:
            # v_feat_inter, t_feat_inter = self.inter_ma(v_feat, t_feat)
            v_feat_inter, t_feat_inter = [], []
            for i in range(n_level):
                v_feat_inter_, t_feat_inter_ = self.inter_ma[i](v_feat[i], t_feat[i])
                v_feat_inter.append(v_feat_inter_)
                t_feat_inter.append(t_feat_inter_)
        else:
            v_feat_inter = [torch.zeros(v_feat_.size(), dtype=torch.float32, device='cuda:0') for v_feat_ in v_feat]
            t_feat_inter = [torch.zeros(t_feat_.size(), dtype=torch.float32, device='cuda:0') for t_feat_ in t_feat]

        # ------------------------Intra-MA-------------------------------
        if self.intra_ma is not None:
            v_feat_intra, t_feat_intra, m_v, m_t = self.intra_ma(v_feat, t_feat)

            m_y = create_elipsis_mask(gt_bboxes, img[0].shape, mode=self.intra_mode)

            # draw_heatmap(m_y, str(self.iteration))
            # import pdb; pdb.set_trace()

            if self.weighted_loss is not None:
                m_v_loss = multi_level_mask_loss(m_v, m_y, w=w_v, mode=self.intra_mode)
                m_t_loss = multi_level_mask_loss(m_t, m_y, w=w_t, mode=self.intra_mode)

                m_loss = m_v_loss + m_t_loss
            else:
                m_v_loss = multi_level_mask_loss(m_v, m_y, mode=self.intra_mode)
                m_t_loss = multi_level_mask_loss(m_t, m_y, mode=self.intra_mode)

                m_loss = (m_v_loss + m_t_loss)/2

            mk_loss = dict(mask_loss=m_loss*self.w_m)
            losses.update(mk_loss)
        else:
            v_feat_intra = [torch.zeros(v_feat_.size(), dtype=torch.float32, device='cuda:0') for v_feat_ in v_feat]
            t_feat_intra = [torch.zeros(t_feat_.size(), dtype=torch.float32, device='cuda:0') for t_feat_ in t_feat]

        # -------------------fusion-------------------------------
        v_feat_hat = []
        t_feat_hat = []
        x = []
        for i in range(len(v_feat)):
            if self.intra_ma is None and self.inter_ma is None:
                v_feat_hat_ = v_feat[i]
                t_feat_hat_ = t_feat[i]
            else:
                v_feat_hat_ = v_feat_inter[i] + v_feat_intra[i]
                t_feat_hat_ = t_feat_inter[i] + t_feat_intra[i]
                
            x_ = v_feat_hat_*w_v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + t_feat_hat_*w_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            v_feat_hat.append(v_feat_hat_)
            t_feat_hat.append(t_feat_hat_)
            x.append(x_)
        # --------------------------------------------------------

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        v_feat, t_feat = self.extract_feat(img)

        n_level = len(v_feat)

        self.iteration += 1

        # from time import time
        # t = time()
        timestamp = str(int(self.iteration))
        # save_feature_to_img(v_feat, 'v_', timestamp, method='cv2')
        # save_feature_to_img(t_feat, 't_', timestamp, method='cv2')

        # -------------illumination-guided weighting module---------------
        if self.igfw is not None:
            w_v, w_t, p_dn, _ = self.igfw(img[0])

            # self.writer.add_scalar('w_v', w_v, self.iteration)
        else:
            w_v = torch.tensor(0.5, dtype=torch.float32, device='cuda:0')
            w_t = torch.tensor(0.5, dtype=torch.float32, device='cuda:0')
        # -------------------------Inter-MA------------------------------
        if self.inter_ma is not None:
            # v_feat_inter, t_feat_inter = self.inter_ma(v_feat, t_feat)
            v_feat_inter, t_feat_inter = [], []
            for i in range(n_level):
                v_feat_inter_, t_feat_inter_ = self.inter_ma[i](v_feat[i], t_feat[i])
                v_feat_inter.append(v_feat_inter_)
                t_feat_inter.append(t_feat_inter_)
        else:
            v_feat_inter = [torch.zeros(v_feat_.size(), dtype=torch.float32, device='cuda:0') for v_feat_ in v_feat]
            t_feat_inter = [torch.zeros(t_feat_.size(), dtype=torch.float32, device='cuda:0') for t_feat_ in t_feat]

        # save_feature_to_img(v_feat_inter, 'v_inter_', timestamp, method='cv2')
        # save_feature_to_img(t_feat_inter, 't_inter_', timestamp, method='cv2')

        # ------------------------Intra-MA-------------------------------
        if self.intra_ma is not None:
            v_feat_intra, t_feat_intra, m_v, m_t = self.intra_ma(v_feat, t_feat)
        else:
            v_feat_intra = [torch.zeros(v_feat_.size(), dtype=torch.float32, device='cuda:0') for v_feat_ in v_feat]
            t_feat_intra = [torch.zeros(t_feat_.size(), dtype=torch.float32, device='cuda:0') for t_feat_ in t_feat]

        # save_feature_to_img(v_feat_intra, 'v_intra_', timestamp, method='cv2')
        # save_feature_to_img(t_feat_intra, 't_intra_', timestamp, method='cv2')
        # save_feature_to_img(m_v, 'heatmap_v', self.iteration, method='matshow')
        # save_feature_to_img(m_t, 'heatmap_t', self.iteration, method='matshow')

        # -------------------fusion-------------------------------
        v_feat_hat = []
        t_feat_hat = []
        x = []
        for i in range(len(v_feat)):
            if self.intra_ma is None and self.inter_ma is None:
                v_feat_hat_ = v_feat[i]
                t_feat_hat_ = t_feat[i]
            else:
                v_feat_hat_ = v_feat_inter[i] + v_feat_intra[i]
                t_feat_hat_ = t_feat_inter[i] + t_feat_intra[i]
            x_ = v_feat_hat_*w_v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + t_feat_hat_*w_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            v_feat_hat.append(v_feat_hat_)
            t_feat_hat.append(t_feat_hat_)
            x.append(x_)

        # save_feature_to_img(v_feat_hat, 'v_hat_', timestamp, method='cv2')
        # save_feature_to_img(t_feat_hat, 't_hat_', timestamp, method='cv2')
        # save_feature_to_img(x, 'fused_', timestamp, method='cv2')

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

def draw_heatmap(heatmap, timestamp):
    import cv2
    import os

    dist_dir = '/home/zhangyan22/mmdetection/work_dir/faster_rcnn_r50_fpn_tinet/flir2/vis'
    dist_dir = os.path.join(dist_dir, timestamp)
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)

    for i in range(heatmap.shape[0]):
        img = heatmap[i,:,:].cpu().numpy()
        img = (img - np.amin(img))/(np.amax(img) - np.amin(img) + 1e-5) * 255 # 注意要防止分母为0！ 
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(dist_dir, f'heatmap_gt_{i}.png'), img)
