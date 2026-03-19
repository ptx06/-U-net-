import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Score(object):
    """计算分割指标（像素准确率、IoU等）"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_scores(self):
        hist = self.confusion_matrix
        pixel_acc = np.diag(hist).sum() / hist.sum()
        class_acc = np.diag(hist) / hist.sum(axis=1)
        mean_class_acc = np.nanmean(class_acc)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            'pixel_acc': pixel_acc,
            'class_acc': class_acc,
            'mean_class_acc': mean_class_acc,
            'mIou': mean_iu,
            'fw_iou': fw_iu,
            'per_class_iou': cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def make_one_hot(input, num_classes):
    """将标签转换为one-hot编码（用于Dice Loss）"""
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(input.device)
    result = result.scatter_(1, input.long(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """二分类Dice Loss"""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        intersection = (predict * target).sum(-1)
        denominator = (predict.pow(self.p) + target.pow(self.p)).sum(-1)
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """多分类Dice Loss"""
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        n_classes = predict.shape[1]
        target_one_hot = make_one_hot(target.unsqueeze(1), n_classes)
        dice_loss = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        for i in range(n_classes):
            if i == self.ignore_index:
                continue
            predict_i = predict[:, i:i+1, ...]
            target_i = target_one_hot[:, i:i+1, ...]
            loss_i = dice_loss(predict_i, target_i)
            if self.weight is not None:
                loss_i *= self.weight[i]
            total_loss += loss_i
        return total_loss / n_classes