import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from dataset import TestDataset
from utils import performances_cross_db
from model.net import EnsembleNet_c3


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


def test_on_dataset(model, epoch, test_loader, test_label, writer, device, cen_criterion, mse_loss):
    model.eval()
    raw_test_scores, gt_labels = [], []
    mean_loss_bce, mean_loss_mse = [], []
    vgg_act, alex_act, label_list = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, label = data["images"].to(device), data["labels"].to(device)

            # mag, phase = torch.chunk(fft, 2, dim=-3)
            combined_score, raw_scores = model(img)

            alex_act.append(combined_score)
            label_list.append(label)

            mean_loss_bce.append(cen_criterion(raw_scores.squeeze(), label).item())
            mean_loss_mse.append(mse_loss(raw_scores.squeeze(), label).item())
            # mean_loss_bce_map.append(cen_criterion(map_output.squeeze(), map_label).item())
            # mean_loss_mse_map.append(mse_loss(map_output.squeeze(), map_label).item())

            # mean_map_scores = torch.mean(map_output, dim=[1, 2, 3]).cpu().data.numpy()
            raw_scores = raw_scores.cpu().data.numpy()
            raw_test_scores.extend(raw_scores)
            gt_labels.extend(label.cpu().data.numpy())

        # show_tSNE(vgg_act, label_list, 'SynthASpoof - alex_1_features')
        # show_tSNE(alex_act, label_list, 'MSU-MFSD - combined_score')

        performances_cross_db(raw_test_scores, gt_labels, writer=writer, epoch=epoch, test_label=test_label)

        # writer.add_image(f'test_{test_label}/images', make_grid(img), epoch)
        # writer.add_image(f'test_{test_label}/magnitude', make_grid(mag), epoch)
        # writer.add_image(f'test_{test_label}/phase', make_grid(phase), epoch)

        writer.add_scalar(f'test_{test_label}/bce_loss', np.mean(mean_loss_bce), epoch)
        writer.add_scalar(f'test_{test_label}/mse_loss', np.mean(mean_loss_mse), epoch)
        # writer.add_scalar(f'test_{test_label}/map_bce_loss', np.mean(mean_loss_bce_map), epoch)
        # writer.add_scalar(f'test_{test_label}/mse_map', np.mean(mean_loss_mse_map), epoch)

        # Add the raw_test_scores to TensorBoard
        fig, ax = plt.subplots()
        ax.plot(raw_test_scores)
        ax.set_xlabel('item')
        ax.set_ylabel('score')
        writer.add_figure(f'test_{test_label}/scores', fig, epoch)

        return gt_labels, raw_test_scores


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    torch.cuda.empty_cache()
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description='SynPAD Training with MixStyle')
    parser.add_argument("--prefix", default='casia-cefa', type=str, help="testing dataset")
    parser.add_argument("--weight_path", default='checkpoints/ckpt.pth', type=str, help="model weights")
    parser.add_argument("--csv_test", default='dataset/CASIA-CEFA/test.csv', type=str, help="csv contains training data")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")

    args = parser.parse_args()

    if not os.path.isdir('logs'):
        os.makedirs('logs')

    writer = SummaryWriter(log_dir='logs/test_{}'.format(args.prefix))

    test_dataset = TestDataset(csv_file=args.csv_test, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = EnsembleNet_c3()
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(args.weight_path))

    # cen_criterion = torch.nn.BCELoss().to(device)
    cen_criterion = torch.nn.BCEWithLogitsLoss().to(device)
    mse_loss = torch.nn.MSELoss().to(device)

    model.eval()
    true_label, prediction_score = test_on_dataset(model, 0, test_loader, args.prefix,
                                                   writer, device, cen_criterion, mse_loss)

    # image_path, true_label, prediction_score, prediction_label
    dataframe = pd.read_csv(args.csv_test)
    image_path = dataframe.iloc[:, 0].values.tolist()
    # true_label = gt_labels
    # prediction_score = raw_test_scores
    prediction_label = [1 if p >= 0 else 0 for p in prediction_score]

    total_df = pd.DataFrame({'image_path': image_path, 'true_label': true_label,
                             'prediction_score': prediction_score,
                             'prediction_label': prediction_label})
    total_df.to_csv('prediction.csv', index=False)
