# -*- coding: utf-8 -*-
# @Time : 2022/9/4 16:45
# @Author : Mengtian Zhang
# @E-mail : zhangmengtian@sjtu.edu.cn
# @Version : v-dev-0.0
# @License : MIT License
# @Copyright : Copyright 2022, Mengtian Zhang
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import argparse
import os

import numpy as np
from torch import optim

from datasets.dataset_cifar_dir import dataloader_cifar_noisy
from datasets import data_loader_clothes1m
from nets import PreResNet
from nets.ResNetWithClusters import MultipleTaskNetwork
from tools import utils
import torch
from nets.losses.loss_dividemix import LossControl
from torchvision.models import resnet50, ResNet50_Weights


def get_args():
    parser = argparse.ArgumentParser(description='TRAIN CIFAR MAIN.')

    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--train-data-percent', default=1.0, type=float, help='Data percent')
    parser.add_argument('--noise_mode', default='worse_label', type=str, help="noise mode")
    parser.add_argument('--noise_rate', default=-1, type=float, help='Noise rate.')

    parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loader workers')

    parser.add_argument('--pretrained', action="store_true", default=False, help="Network with pretrained weights")
    parser.add_argument('--net', default='pre_resnet34', type=str, help="Network name")

    parser.add_argument('--epochs', default=300, type=int)
    # parser.add_argument('--epochs_warmup', default=10, type=int)
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--num_class', default=10, type=int)

    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')

    # parser.add_argument('--cluster_num', default=500, type=int)
    # parser.add_argument('--cluster_interval', default=30, type=int)

    parser.add_argument('--num_batches', default=1000, type=int)
    parser.add_argument('--num_samples', default=-1, type=int)

    _args = parser.parse_args()

    return _args


def create_network(net_name='pre_resnet34', num_classes=10, z_dim=128, n_clusters=10, norm_p=2, normalize=True):
    _net = MultipleTaskNetwork(net_name, num_classes=num_classes, z_dim=z_dim,
                               n_clusters=n_clusters, norm_p=norm_p, normalize=normalize)
    return _net


def create_network_for_clothes1m(net_name='resnet50', num_classes=14, z_dim=128, n_clusters=10, norm_p=2, normalize=True):
    pretrained_parameters = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    _net = MultipleTaskNetwork(net_name, num_classes=num_classes, z_dim=z_dim,
                               n_clusters=n_clusters, norm_p=norm_p, normalize=normalize)
    _net.backbone.load_state_dict(pretrained_parameters, strict=False)

    return _net


# Parameters
args = get_args()
print(args)
utils.seed_everything(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Paths
output_dir = os.path.join("./Outputs", utils.get_day_stamp(), f"{args.dataset}-{args.num_samples}",
                          'TrainNoisy', f"class-num-{args.num_class}", args.noise_mode,
                          f"data-{args.train_data_percent:.2f}"
                          f"_noise-{args.noise_rate:.2f}"
                          f"_{args.net}_epochs-{args.epochs}"
                          f"_seed-{args.seed}_"
                          f"{utils.get_timestamp()}"
                          )
utils.makedirs(output_dir)
args.output_dir = output_dir
utils.save_args(args, path=os.path.join(output_dir, 'args.txt'))

# Initialization
if args.dataset == 'cifar10':
    dataset_getter = dataloader_cifar_noisy.generate_cifar_n_dataset_getter_dividemix(
        data_dir='./data/cifar-10-batches-py',
        index_root='./data/cifar-n-processed/cifar-10/2022_9_3_10_51_27',
        data_percent=args.train_data_percent,
        noise_mode=args.noise_mode,
        noise_label_sample_rate=args.noise_rate
    )
    net_1 = create_network(net_name=args.net, num_classes=args.num_class).to(device)
    net_2 = create_network(net_name=args.net, num_classes=args.num_class).to(device)

    from Modules.dividemix_cifar import DivideMixTrainCifar as TrainModule

elif args.dataset == 'cifar100':
    dataset_getter = dataloader_cifar_noisy.generate_cifar_100n_dataset_getter(
        data_dir='./data/cifar-100-python',
        noise_path='./data/cifar-n-labels/CIFAR-100_human.pt'
    )
    net_1 = create_network(net_name=args.net, num_classes=args.num_class).to(device)
    net_2 = create_network(net_name=args.net, num_classes=args.num_class).to(device)

    from Modules.dividemix_cifar import DivideMixTrainCifar as TrainModule

elif args.dataset == 'clothes1m':
    dataset_getter = data_loader_clothes1m.generate_clothes1m_dataset_getter(num_samples=args.num_samples)
    net_1 = create_network_for_clothes1m().to(device)
    net_2 = create_network_for_clothes1m().to(device)

    from Modules.dividemix_clothes1m import DivideMixTrainCifar as TrainModule

else:
    raise NotImplementedError

# dataset_getter.set_label_noisy(np.load('./Outputs/2022_9_5/evaluations/label_modified.npy'))
loss_controller = LossControl(lambda_u=args.lambda_u)

print('| Building net')

optimizer_1 = optim.SGD(net_1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_2 = optim.SGD(net_2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Main
train_module = TrainModule(net_1=net_1, net_2=net_2, optimizer_1=optimizer_1, optimizer_2=optimizer_2,
                           dataset_getter=dataset_getter, loss_controller=loss_controller,
                           device=device, args=args)

train_module.train_noisy(epochs=args.epochs)
