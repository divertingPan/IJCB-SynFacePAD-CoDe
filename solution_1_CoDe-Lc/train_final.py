import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import TrainDataset, TestDataset, ApplyWeightedRandomSampler
from utils import performances_cross_db
from model.net import MultiFTNet, EnsembleNet, EnsembleNet_c4
from test import test_on_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # cv2.setNumThreads(0)
    # cv2.ocl.setUseOpenCL(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    torch.cuda.empty_cache()
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description='SynPAD Training with MixStyle')
    parser.add_argument("--prefix", default='contrast_4_relu_tanh', type=str, help="log description")
    parser.add_argument("--model_name", default='alex_alex', type=str, help="model backbone")
    parser.add_argument("--csv_train", default='dataset/SynthASpoof/train.csv', type=str,
                        help="csv contains training data")

    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--max_epoch", default=200, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")
    parser.add_argument("--log_step", default=50, type=int, help="log step")

    args = parser.parse_args()

    if not os.path.isdir('logs'):
        os.makedirs('logs')

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(args.model_name, args.prefix))

    # WeightedRandomSampler to balance the attack and bonafide in a mini-batch
    train_dataset_1 = TrainDataset(csv_file=args.csv_train, input_shape=args.input_shape)
    train_loader_1 = DataLoader(train_dataset_1, batch_size=args.batch_size,
                                sampler=ApplyWeightedRandomSampler(args.csv_train),
                                num_workers=4, pin_memory=True, drop_last=True)

    train_dataset_2 = TrainDataset(csv_file=args.csv_train, input_shape=args.input_shape)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size,
                                sampler=ApplyWeightedRandomSampler(args.csv_train),
                                num_workers=4, pin_memory=True, drop_last=True)

    # test_dataset = TestDataset(csv_file='dataset/SynthASpoof/test.csv', input_shape=args.input_shape)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=4, pin_memory=True)
    #
    # test_dataset_fasd = TestDataset(csv_file='dataset/CASIA-FASD/test.csv', input_shape=args.input_shape)
    # test_loader_fasd = DataLoader(test_dataset_fasd, batch_size=args.batch_size,
    #                               shuffle=False, num_workers=4, pin_memory=True)
    #
    # test_dataset_cefa = TestDataset(csv_file='dataset/CASIA-CEFA/test.csv', input_shape=args.input_shape)
    # test_loader_cefa = DataLoader(test_dataset_cefa, batch_size=args.batch_size,
    #                               shuffle=False, num_workers=4, pin_memory=True)
    #
    # test_dataset_msu = TestDataset(csv_file='dataset/MSU-MFSD/test.csv', input_shape=args.input_shape)
    # test_loader_msu = DataLoader(test_dataset_msu, batch_size=args.batch_size,
    #                              shuffle=False, num_workers=4, pin_memory=True)

    checkpoint_save_dir = os.path.join('checkpoints', '{}_{}'.format(args.model_name, args.prefix))
    print('Checkpoint folder', checkpoint_save_dir)
    if not os.path.isdir(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)

    model = EnsembleNet_c4()
    # model = torch.nn.DataParallel(model)
    model = model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    # cen_criterion = torch.nn.BCELoss().to(device)
    cen_criterion = torch.nn.BCEWithLogitsLoss().to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    feature_mse_loss = torch.nn.MSELoss(reduction='none').to(device)
    cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

    for epoch in range(args.max_epoch):
        if os.path.isfile(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))):
            model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))))
            continue
        else:
            print('-------------- train ------------------------')
            print('epoch: {}  lr: {:.8f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

            model.train()
            num_iter = len(train_loader_1)
            for i, (data_1, data_2) in enumerate(zip(train_loader_1, train_loader_2)):
                img_1, label_1 = data_1["images"].to(device), data_1["labels"].to(device)
                img_2, label_2 = data_2["images"].to(device), data_2["labels"].to(device)

                # mag, phase = torch.chunk(fft, 2, dim=-3)

                vgg_features_1, a_features_1, vgg_score_1, alex_score_1, score_1 = model(img_1)
                vgg_features_2, a_features_2, vgg_score_2, alex_score_2, score_2 = model(img_2)

                loss_bce_cls = (cen_criterion(score_1.squeeze(), label_1) +
                                cen_criterion(score_2.squeeze(), label_2))
                loss_ascore_cls = (cen_criterion(alex_score_1.squeeze(), label_1) +
                                   cen_criterion(alex_score_2.squeeze(), label_2))
                loss_vscore_cls = (cen_criterion(vgg_score_1.squeeze(), label_1) +
                                   cen_criterion(vgg_score_2.squeeze(), label_2))
                # loss_bce_alex = cen_criterion(alex_o.squeeze(), label)
                # loss_bce_vgg = cen_criterion(vgg_o.squeeze(), label)
                # loss_bce_fas = cen_criterion(fas_o.squeeze(), label)
                # loss_mse = mse_loss(cls.squeeze(), label)
                # loss_fft = mse_loss(feature_fft, fft) + mse_loss(feature_fft_2, fft_2)

                label_flag = (-1) ** (label_1 + label_2)
                loss_v_cossim = (1 - cossim(vgg_features_1, vgg_features_2) * label_flag).mean()
                loss_v_feature = (feature_mse_loss(vgg_features_1, vgg_features_2).mean(dim=1) * label_flag).mean()

                loss_a_cossim = (1 - cossim(a_features_1, a_features_2) * label_flag).mean()
                loss_a_feature = (feature_mse_loss(a_features_1, a_features_2).mean(dim=1) * label_flag).mean()

                loss = (loss_bce_cls + loss_ascore_cls + loss_vscore_cls +
                        loss_v_cossim + loss_v_feature + loss_a_cossim + loss_a_feature)

                # loss_bce = cen_criterion(output.squeeze(), label)
                # loss_mse = mse_loss(output.squeeze(), label)
                # loss_bce_map = cen_criterion(map_output.squeeze(), map_label)
                # loss_mse_map = mse_loss(map_output.squeeze(), map_label)

                if (i + 1) % args.log_step == 0:
                    flags = num_iter * epoch + i
                    print(f'Iter [{i + 1}/{num_iter}]'
                          f' Loss_cls: {loss_bce_cls.item():.4f},'
                          f' Loss_alex: {loss_ascore_cls.item():.4f},'
                          f' Loss_vgg: {loss_vscore_cls.item():.4f},'
                          f' Loss_v_cos: {loss_v_cossim.item():.4f},'
                          f' Loss_v_mse: {loss_v_feature.item():.4f},'
                          f' Loss_a_cos: {loss_a_cossim.item():.4f},'
                          f' Loss_a_mse: {loss_a_feature.item():.4f},')

                    writer.add_scalar('train_ensemble/cls_loss', loss_bce_cls.item(), flags)
                    writer.add_scalar('train_ensemble/alex_loss', loss_ascore_cls.item(), flags)
                    writer.add_scalar('train_ensemble/vgg_loss', loss_vscore_cls.item(), flags)
                    writer.add_scalar('train_ensemble/v_cos_loss', loss_v_cossim.item(), flags)
                    writer.add_scalar('train_ensemble/v_mse_loss', loss_v_feature.item(), flags)
                    writer.add_scalar('train_ensemble/a_cos_loss', loss_a_cossim.item(), flags)
                    writer.add_scalar('train_ensemble/a_mse_loss', loss_a_feature.item(), flags)

                    # writer.add_image('train/image', make_grid(img), flags)
                    # writer.add_image('train/magnitude', make_grid(mag), flags)
                    # writer.add_image('train/phase', make_grid(phase), flags)
                    # break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f'{epoch}.pth'))
            lr_scheduler.step()

        # print('------------------- test SynthASpoof -------------------')
        # test_on_dataset(model, epoch, test_loader, 'SynthASpoof', writer, device, cen_criterion, mse_loss)
        #
        # print('------------------- test CASIA-FASD -------------------')
        # test_on_dataset(model, epoch, test_loader_fasd, 'CASIA-FASD', writer, device, cen_criterion, mse_loss)
        #
        # print('------------------- test CASIA-CEFA -------------------')
        # test_on_dataset(model, epoch, test_loader_cefa, 'CASIA-CEFA', writer, device, cen_criterion, mse_loss)
        #
        # print('------------------- test MSU-MFSD -------------------')
        # test_on_dataset(model, epoch, test_loader_msu, 'MSU-MFSD', writer, device, cen_criterion, mse_loss)

        # break
