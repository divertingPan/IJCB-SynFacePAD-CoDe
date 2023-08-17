'''
Evaluation code is based on 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' By Zitong Yu & Zhuo Su, 2019
github: https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/utils.py
'''

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from collections import defaultdict


def hypersphere_loss(feature, label, r=2, m=10, device='cpu'):
    """
    :param device: cpu or cuda
    :param feature: feature
    :param label: true label, Bonafide=1, attack=0, shape: [128]
    :param r: r
    :param m: m
    :return: loss
    """
    batchsize = feature.shape[0]
    ds = (torch.ones(batchsize,) * (r ** 2)).to(device)
    dl = (torch.ones(batchsize,) * ((r + m) ** 2)).to(device)  # shape: [128]
    d = torch.norm(feature, p=2, dim=1) ** 2  # shape: [128]

    ln = torch.max(torch.zeros(batchsize, ).to(device), d - ds) * label
    la = torch.max(torch.zeros(batchsize, ).to(device), dl - d) * (1 - label)

    # print("d: ", d[:10])
    # print("label: ", label[:10])
    # print("ln: ", ln[:10])
    # print("la: ", la[:10])

    loss = torch.mean(ln + la)

    return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_err_threhold_cross_db(fpr, tpr, threshold):
    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th, right_index


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', writer=None, epoch=0, test_label=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="yellow" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    writer.add_figure(f'test_{test_label}/confusion_matrix', fig, epoch)


def performances_cross_db(prediction_scores, gt_labels, pos_label=1, verbose=True, writer=None, epoch=0, test_label=None):
    # data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=pos_label)

    precision, recall, _ = precision_recall_curve(gt_labels, prediction_scores, pos_label=pos_label)
    pr_auc = auc(recall, precision)

    val_err, val_threshold, right_index = get_err_threhold_cross_db(fpr, tpr, threshold)
    roc_auc = auc(fpr, tpr)

    FRR = 1 - tpr  # FRR = 1 - TPR
    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    predicted_classes = [1 if p >= 0 else 0 for p in prediction_scores]
    correct_predictions = [1 if p == t else 0 for p, t in zip(predicted_classes, gt_labels)]
    Acc = sum(correct_predictions) / len(correct_predictions)
    # Acc = (sum([abs(prediction_scores[i] - gt_labels[i]) < 0.5 for i in range(len(gt_labels))]) / len(gt_labels))[0]

    # Calculate the confusion matrix
    cm = confusion_matrix(gt_labels, predicted_classes)

    if verbose is True:
        print(f'AUC@ROC: {roc_auc:.4f}, HTER: {HTER[right_index]:.4f}, APCER: {fpr[right_index]:.4f},'
              f' BPCER: {FRR[right_index]:.4f}, EER: {val_err:.4f}, TH: {val_threshold}, Acc: {Acc:.4f}')

    if writer:
        writer.add_scalar(f'test_{test_label}/Acc', Acc, epoch)
        writer.add_scalar(f'test_{test_label}/AUC', roc_auc, epoch)
        writer.add_scalar(f'test_{test_label}/HTER', HTER[right_index], epoch)
        writer.add_scalar(f'test_{test_label}/APCER', fpr[right_index], epoch)
        writer.add_scalar(f'test_{test_label}/BPCER', FRR[right_index], epoch)
        writer.add_scalar(f'test_{test_label}/threshold', val_threshold, epoch)

        # Add the precision-recall curve to TensorBoard
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        writer.add_figure(f'test_{test_label}/pr_curve', fig, epoch)

        # Add the ROC curve to TensorBoard
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        writer.add_figure(f'test_{test_label}/roc_curve', fig, epoch)

        # Add the confusion matrix to TensorBoard
        classes = ['PA', 'Real']  # Specify the class names
        plot_confusion_matrix(cm, classes=classes, writer=writer, epoch=epoch, test_label=test_label)

    return roc_auc, fpr[right_index], FRR[right_index], HTER[right_index], Acc


def compute_video_score(video_ids, predictions, labels):
    predictions_dict, labels_dict = defaultdict(list), defaultdict(list)

    for i in range(len(video_ids)):
        video_key = video_ids[i]
        predictions_dict[video_key].append(predictions[i])
        labels_dict[video_key].append(labels[i])

    new_predictions, new_labels, new_video_ids = [], [], []

    for video_indx in list(set(video_ids)):
        new_video_ids.append(video_indx)
        scores = np.mean(predictions_dict[video_indx])

        label = labels_dict[video_indx][0]
        new_predictions.append(scores)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids
